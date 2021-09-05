#include <fstream>
#include <string>
#include <vector>
#include <experimental/filesystem>

#include "yaml-cpp/yaml.h"

#include "include/video_data_loader.h"
#include "include/inference_server.h"
#include "include/video_experiment_server.h"

// Expects a validation directory as in pytorch
std::vector<std::string> GetFileNames(const std::string& vid_dir) {
  namespace fs = std::experimental::filesystem;

  std::vector<std::string> str_paths;
  for (size_t i = 0; i < 1; i++) {
  std::vector<fs::path> paths;
  std::copy(fs::directory_iterator(vid_dir),
            fs::directory_iterator(),
            std::back_inserter(paths));
  std::sort(paths.begin(), paths.end(),
            [](const fs::path& pa, const fs::path& pb) {
              std::stringstream sa(pa.filename()), sb(pb.filename());
              int ia, ib;
              sa >> ia; sb >> ib;
              return ia < ib;
            });

  for (auto f : paths)
    str_paths.push_back(f.string());
  }
  // str_paths.erase(str_paths.begin() + 1000, str_paths.end());
  return str_paths;
}

int main(int argc, char *argv[]) {
  // auto paths = GetFileNames("/lfs/1/ddkang/blazeit/data/svideo/jackson-town-square/2017-12-17");
  // auto paths = GetFileNames("/lfs/1/ddkang/blazeit/data/svideo/jackson-town-square/short");
  // auto paths = GetFileNames("/lfs/1/ddkang/vision-inf/data/svideo/decode-test/240p");

  assert(argc == 2);
  YAML::Node cfg = YAML::LoadFile(argv[1]);
  // Record the config
  std::cout << cfg << std::endl << std::endl;

  const bool kWriteOut = cfg["experiment-config"]["write-out"].as<bool>();
  const bool kRunInfer = cfg["experiment-config"]["run-infer"].as<bool>();
  const bool kDoMemcpy = cfg["infer-config"]["do-memcpy"].as<bool>();

  // Video only has one model
  auto model_cfg = cfg["model-config"]["model-single"];
  const std::string kOnnxPath = model_cfg["onnx-path"].as<std::string>();
  const std::string kEnginePath = model_cfg["engine-path"].as<std::string>();
  const std::string kDataPath =  model_cfg["data-path"].as<std::string>();
  const size_t kBatchSize = model_cfg["batch-size"].as<size_t>();
  const size_t kModelInputDim = model_cfg["input-dim"][0].as<size_t>();

  auto paths = GetFileNames(kDataPath);
  std::cerr << "Processing " << paths.size() << " files" << std::endl;

  auto crop_cfg = cfg["crop"];
  const size_t xmin = crop_cfg["xmin"].as<size_t>();
  const size_t ymin = crop_cfg["ymin"].as<size_t>();
  const size_t xmax = crop_cfg["xmax"].as<size_t>();
  const size_t ymax = crop_cfg["ymax"].as<size_t>();
  const CropRegion region(xmin, ymin, xmax, ymax);

  // FIXME: 256
  VideoDataLoader *loader = NULL;
  const std::string kLoaderType = model_cfg["data-loader"].as<std::string>();
  std::string cond_str = cfg["experiment-config"]["exp-type"].as<std::string>();
  LoaderCondition cond = LoaderCondition::GetVal(cond_str);
  if (kLoaderType == "opt") {
    loader = new OptimizedVidDataLoader(256, kModelInputDim, region, cond);
  } else if (kLoaderType == "naive") {
    loader = new NaiveVidDataLoader(256, kModelInputDim, region, cond);
  } else {
    throw std::invalid_argument("Loader cfg wrong");
  }
  OnnxInferenceServer *infer;
  namespace fs = std::experimental::filesystem;
  if (fs::exists(kEnginePath)) {
    infer = new OnnxInferenceServer(kEnginePath, kBatchSize, kDoMemcpy);
  } else {
    infer = new OnnxInferenceServer(
        kOnnxPath, "", kEnginePath, kBatchSize, kDoMemcpy,
        loader, paths,
        model_cfg["do-int8"].as<bool>());
  }
  VideoExperimentServer server(*loader, infer, kBatchSize, kRunInfer);

  infer->warmup(kModelInputDim);

  float time;
  std::vector<float> output;
  std::tie(time, output) = server.TimeEndToEnd(paths);
  std::cerr << "Runtime: " << time << std::endl;

  if (kWriteOut) {
    std::ofstream fout("preds.out", std::ios::out | std::ios::binary);
    fout.write((char *) output.data(), output.size() * sizeof(float));
    fout.close();
  }

  return 0;
}
