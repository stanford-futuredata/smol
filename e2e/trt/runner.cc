#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <experimental/filesystem>

#include "yaml-cpp/yaml.h"

#include "include/data_loader.h"
#include "include/inference_server.h"
#include "include/experiment_server.h"
#include "include/criterion.h"

// Expects a validation directory as in pytorch
std::vector<std::string> GetFileNames(const std::string& val_dir) {
  namespace fs = std::experimental::filesystem;

  std::vector<std::string> file_paths;
  std::vector<fs::path> dirs;
  std::copy(fs::directory_iterator(val_dir), fs::directory_iterator(), std::back_inserter(dirs));
  std::sort(dirs.begin(), dirs.end());

  for (const auto& dir : dirs) {
    if (fs::is_directory(dir)) {
      std::vector<fs::path> fnames;
      std::copy(fs::directory_iterator(dir.string()), fs::directory_iterator(), std::back_inserter(fnames));
      std::sort(fnames.begin(), fnames.end());
      for (const auto& fname : fnames) {
        std::string file_name = fname.string();
        file_paths.push_back(file_name);
      }
    }
  }

  return file_paths;
}

std::vector<CompressedImage> GetCompressed(
    const std::vector<std::string>& file_paths,
    const DataLoader& loader,
    const size_t kMult) {
  std::vector<CompressedImage> ret(file_paths.size());
  #pragma omp parallel for
  for (size_t i = 0; i < file_paths.size(); i++) {
    ret[i] = loader.LoadCompressedImageFromFile(file_paths[i]);
  }
  for (size_t k = 0; k < kMult - 1; k++) {
    for (size_t i = 0; i < file_paths.size(); i++)
      ret.push_back(ret[i]);
  }
  return ret;
}

class InferenceConfig {
 public:
  std::string kDataPath_;
  const size_t kBatchSize_, kModelInputDim_;
  DataLoader *loader;
  OnnxInferenceServer *infer;

  InferenceConfig(
      const std::string& kDataPath,
      const std::string& kEnginePath,
      const std::string& kOnnxPath, const std::string& kOnnxPathBS1,
      const size_t kBatchSize, const size_t kModelInputDim,
      const size_t kDoMemcpy,
      DataLoader *loader,
      const bool kDoResize = true,
      const bool kDoINT8 = false,
      const bool kDoWarmup = true) :
      kDataPath_(kDataPath),
      kBatchSize_(kBatchSize), kModelInputDim_(kModelInputDim),
      loader(loader) {
    namespace fs = std::experimental::filesystem;
    if (fs::exists(kEnginePath)) {
      infer = new OnnxInferenceServer(kEnginePath, kBatchSize, kDoMemcpy);
    } else {
      std::vector<CompressedImage> compressed(0);
      if (kDoINT8) {
        compressed = GetCompressed(GetFileNames(kDataPath), *loader, 1);
      }
      infer = new OnnxInferenceServer(
          kOnnxPath, kOnnxPathBS1, kEnginePath,
          kBatchSize, kDoMemcpy,
          loader, compressed,
          kDoINT8, !kDoResize);
      for (size_t i = 0; i < compressed.size(); i++) {
        free(compressed[i].first);
      }
      compressed.erase(compressed.begin(), compressed.end());
    }
    if (kDoWarmup) {
      warmup();
    }
  }

  void warmup() {
    infer->warmup(kBatchSize_);
  }
};

static std::vector<size_t> MaskToIndMap(const std::vector<bool>& mask) {
  const size_t kNNZ = std::accumulate(mask.begin(), mask.end(), (size_t) 0);
  std::cerr << "kNNZ: " << kNNZ << std::endl;
  std::vector<size_t> ret(kNNZ);
  for (size_t mask_idx = 0, ret_idx = 0; mask_idx < mask.size(); mask_idx++) {
    if (mask[mask_idx]) {
      ret[ret_idx++] = mask_idx;
    }
  }

  return ret;
}

int main(int argc, char *argv[]) {
  // auto paths = GetFileNames("/lfs/1/ddkang/vision-inf/data/imagenet/val/");
  // auto paths = GetFileNames("/lfs/1/ddkang/vision-inf/data/in-small-jpeg-75/val/");
  // auto paths = GetFileNames("/lfs/1/ddkang/vision-inf/data/in-small-jpeg-95/val/");
  // auto paths = GetFileNames("/lfs/1/ddkang/vision-inf/data/in-small-png/val/");

  assert(argc == 2);
  YAML::Node cfg = YAML::LoadFile(argv[1]);
  std::cout << "Using config file: " << argv[1] << std::endl;
  std::cout << "Config is:" << std::endl;
  // Record the config
  std::cout << cfg << std::endl << std::endl;

  const bool kTimeLoad = cfg["experiment-config"]["time-load"].as<bool>();
  const bool kWriteOut = cfg["experiment-config"]["write-out"].as<bool>();
  const bool kRunInfer = cfg["experiment-config"]["run-infer"].as<bool>();
  const bool kDoMemcpy = cfg["infer-config"]["do-memcpy"].as<bool>();
  const size_t kMult = cfg["experiment-config"]["multiplier"].as<size_t>();

  std::vector<InferenceConfig> configs;
  auto model_cfg = cfg["model-config"];
  std::string cond_str = cfg["experiment-config"]["exp-type"].as<std::string>();
  LoaderCondition cond = LoaderCondition::GetVal(cond_str);
  for (auto it = model_cfg.begin(); it != model_cfg.end(); it++) {
    auto cfg_single = it->second;
    const size_t kResizeDim = cfg_single["resize-dim"] ?
        cfg_single["resize-dim"][0].as<size_t>() : 256;
    const size_t kModelInputDim = cfg_single["input-dim"][0].as<size_t>();
    const std::string loader_type = cfg_single["data-loader"].as<std::string>();
    const bool kDoResize = cfg_single["do-resize"] ? cfg_single["do-resize"].as<bool>() : true;

    DataLoader *loader;
    if (loader_type == "naive") {
      loader = new NaiveDataLoader(kResizeDim, kModelInputDim, kDoResize, cond);
    } else if (loader_type == "opt-jpg") {
      loader = new OptimizedDataLoader(kResizeDim, kModelInputDim, kDoResize, cond);
    } else if (loader_type == "png") {
      loader = new PNGDataLoader(256, kModelInputDim, kDoResize, cond);
    } else if (loader_type == "opt-png") {
      loader = new OptResizePNGDataLoader(kResizeDim, kModelInputDim, kDoResize, cond);
    } else {
      throw std::invalid_argument("Wrong loader type");
    }

    configs.push_back(
        InferenceConfig(
            cfg_single["data-path"].as<std::string>(),
            cfg_single["engine-path"].as<std::string>(),
            cfg_single["onnx-path"].as<std::string>(),
            cfg_single["onnx-path-bs1"].as<std::string>(),
            cfg_single["batch-size"].as<int>(),
            kModelInputDim,
            kDoMemcpy,
            loader,
            kDoResize,
            cfg_single["do-int8"].as<bool>()));
  }

  if (cfg["experiment-type"].as<std::string>() != "full") {
    ExperimentServer server(*configs[0].loader, configs[0].infer,
                            configs[0].kBatchSize_, kRunInfer);
    float time = server.TimeInferenceOnly();
    std::cerr << "Runtime: " << time << std::endl;
    return 0;
  }

  // Set the cascade criterion
  Criterion *criterion = NULL;
  auto crit_string = cfg["criterion"]["name"].as<std::string>();
  if (crit_string == "max") {
    criterion = new MaxCriterion(cfg["criterion"]["param"].as<float>());
  } else if (crit_string == "none") {
  } else {
    throw std::invalid_argument("Criterion wrong");
  }

  // FIXME: cascades are implemented in a really annoying way right now
  std::vector<float> final_output;
  std::vector<bool> mask;
  std::vector<size_t> ind_map;
  for (size_t i = 0; i < configs.size(); i++) {
    InferenceConfig *config = &configs[i];
    const size_t kOutputSingle = config->infer->GetOutputSingle();
    auto base_paths = GetFileNames(config->kDataPath_);
    std::vector<std::string> paths(base_paths.size());
    if (i == 0) {
      paths = base_paths;
    } else {
      auto it = std::copy_if(
          base_paths.begin(), base_paths.end(), paths.begin(),
          [&](const std::string& str) -> bool {
            size_t idx = &str - &base_paths[0];
            return mask[idx];
          });
      paths.resize(std::distance(paths.begin(), it));
    }
    std::cerr << "Paths: " << paths.size() << std::endl;
    ExperimentServer server(*config->loader, config->infer,
                            config->kBatchSize_, kRunInfer);
    float time;
    std::vector<float> output;
    if (kTimeLoad) {
      throw std::runtime_error("Loading not implemented");
      time = server.TimeEndToEnd(paths);
      std::cerr << "Runtime: " << time << std::endl;
    } else {
      auto compressed_images = GetCompressed(paths, *config->loader, kMult);
      std::cerr << "Loaded files from disk\n";
      std::tie(time, output) = server.TimeNoLoad(compressed_images);
      std::cerr << "Runtime: " << time << std::endl;
    }

    // Cascades
    if (i == 0) {
      assert(output.size() % kOutputSingle == 0);
      std::copy(output.begin(), output.end(), std::back_inserter(final_output));
      // FIXME: do softmax anyway?
      if (criterion == NULL)
        continue;
      mask = criterion->filter(output.size() / kOutputSingle, output);
      ind_map = MaskToIndMap(mask);
    } else {
      if (ind_map.size() == 0)
        continue;
      assert(output.size() % kOutputSingle == 0);
      for (size_t i = 0; i < ind_map.size(); i++) {
        const auto kInStart = output.begin() + i * kOutputSingle;
        const auto kInEnd = output.begin() + (i + 1) * kOutputSingle;
        auto kOutStart = final_output.begin() + ind_map[i] * kOutputSingle;
        std::copy(kInStart, kInEnd, kOutStart);
      }
    }
  }

  if (kWriteOut) {
    std::ofstream fout("preds.out", std::ios::out | std::ios::binary);
    fout.write((char *) final_output.data(), final_output.size() * sizeof(float));
    fout.close();
  }

  return 0;
}
