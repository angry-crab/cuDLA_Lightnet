#include <config_parser.h>
#include <assert.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>

DEFINE_string(engine, "lightnet-tiny.bin",
              "engine Path, "
              "engine Path");

DEFINE_bool(dont_show, false,
	    "[Optional] Flag to off screen");

DEFINE_string(d, "",
              "Directory Path, "
              "Directory Path");

DEFINE_string(v, "",
              "Video Path, "
              "Video Path");

DEFINE_int64(cam, -1, "Camera ID");


DEFINE_string(precision, "FP32",
              "[OPTIONAL] Inference precision. Choose from kFLOAT, kHALF and kINT8.");


DEFINE_uint64(batch_size, 1, "[OPTIONAL] Batch size for the inference engine.");
DEFINE_uint64(width, 0, "[OPTIONAL] width for the inference engine.");
DEFINE_uint64(height, 0, "[OPTIONAL] height for the inference engine.");
DEFINE_int64(dla, -1, "[OPTIONAL] DLA");
DEFINE_uint64(c, 8, "[OPTIONAL] num of classes for the inference engine.");

DEFINE_bool(prof, false,
            "[OPTIONAL] Flag to profile layer by layer");

DEFINE_string(rgb, "not-specified",
              "[OPTIONAL] Path to colormap for detections");

DEFINE_string(mask, "../data/t4-seg.colormap",
              "[OPTIONAL] Path to colormap for semantic segmentation"
              "");

DEFINE_string(names, "../data/t4.names",
              "List of names for detections"
              "List of names for detections");

DEFINE_double(thresh, 0.2, "[OPTIONAL] thresh");


DEFINE_double(scale, 1.0, "[OPTIONAL] scale");

DEFINE_bool(cuda, false,
            "[OPTIONAL] Flag to CUDA Preprocessing");

DEFINE_bool(sparse, false,
            "[OPTIONAL] Flag to 2:4 Structured Sparsity");

DEFINE_bool(first, false,
            "[OPTIONAL] Flag to keep high precision for first layer");

DEFINE_bool(last, false,
            "[OPTIONAL] Flag to keep high precision for last layer");

DEFINE_string(dump, "not-specified",
              "[OPTIONAL] Path to dump predictions for mAP calculation");

DEFINE_string(output, "not-specified",
              "[OPTIONAL] Path to dump outputs for pseudo labeling");

DEFINE_double(clip, 0.0, "[OPTIONAL] Clip value for implicit quantization in output");

DEFINE_bool(save_detections, false,
            "[OPTIONAL] Flag to save images overlayed with objects detected.");
DEFINE_string(save_detections_path, "",
              "[OPTIONAL] Path where the images overlayed with bounding boxes are to be saved");

DEFINE_uint64(wx, 0, "[OPTIONAL] position x for display window");
DEFINE_uint64(wy, 0, "[OPTIONAL] position y for display window");
DEFINE_uint64(ww, 0, "[OPTIONAL] width for display window");
DEFINE_uint64(wh, 0, "[OPTIONAL] height for display window");

DEFINE_string(anchors, "not-specified",
              "Anchor size");
DEFINE_int64(num_anchors, 3, "Number of Anchors");

//For subnet (Optional)
DEFINE_string(subnet_engine, "not-specified",
              "Subnet engine Path, "
              "Subnet engine Path");

DEFINE_string(subnet_names, "not-specified",
              "Subnet list of names for detections"
              "Subnet list of names for detections");

DEFINE_string(subnet_anchors, "not-specified",
              "Anchor size");
DEFINE_int64(subnet_num_anchors, 3, "Number of Anchors");

DEFINE_uint64(subnet_c, 2, "[OPTIONAL] num of classes for the subnet inference engine.");

DEFINE_string(subnet_rgb, "not-specified",
              "[OPTIONAL] Path to colormap for detections");

DEFINE_string(target_names, "not-specified",
              "Subnet list of names for detections"
              "Subnet list of names for detections");

DEFINE_string(bluron, "not-specified",
              "Subnet list of names for detections"
              "Subnet list of names for detections");

DEFINE_string(debug_tensors, "not-specified",
              "tensor names for debug");

DEFINE_bool(save_debug_tensors, false,
              "save debug tensors");

std::string
get_engine_path(void)
{
  return FLAGS_engine;
}

std::string
get_directory_path(void)
{
  return FLAGS_d;
}

int
get_camera_id(void)
{
  return FLAGS_cam;
}

std::string
get_video_path(void)
{
  return FLAGS_v;
}

std::string
get_precision(void)
{
  return FLAGS_precision;
}

bool
is_dont_show(void)
{
  return FLAGS_dont_show;
}

bool
get_prof_flg(void)
{
  return FLAGS_prof;
}

int
get_batch_size(void)
{
  return FLAGS_batch_size;
}

int
get_width(void)
{
  return FLAGS_width;
}

int
get_height(void)
{
  return FLAGS_height;
}

int
get_classes(void)
{
  return FLAGS_c;
}

int
get_dla_id(void)
{
  return FLAGS_dla;
}

static void leftTrim(std::string& s)
{
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

static std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

static bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

static std::vector<std::string> loadListFromTextFile(const std::string filename)
{
  assert(fileExists(filename, true));
    std::vector<std::string> list;

    std::ifstream f(filename);
    if (!f)
    {
        std::cout << "failed to open " << filename;
        assert(0);
    }

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;

        else
            list.push_back(trim(line));
    }

    return list;
}

std::vector<std::vector<int>>
get_colormap(void)
{
  std::string filename = FLAGS_rgb;
  std::vector<std::vector<int>> colormap;
  if (filename != "not-specified") {
    std::vector<std::string> color_list = loadListFromTextFile(filename);    
    for (int i = 0; i < (int)color_list.size(); i++) {
      std::string colormapString = color_list[i];
      std::vector<int> rgb;
      while (!colormapString.empty()) {
	size_t npos = colormapString.find_first_of(',');
	if (npos != std::string::npos) {
	  int colormap = (int)std::stoi(trim(colormapString.substr(0, npos)));
	  rgb.push_back(colormap);
	  colormapString.erase(0, npos + 1);
	} else {
	  int colormap = (int)std::stoi(trim(colormapString));
	  rgb.push_back(colormap);
	  break;
	}      
      }
      colormap.push_back(rgb);
    }
  }
  return colormap;
}

std::vector<cudla_lightnet::Colormap>
get_seg_colormap(void)
{
  std::string filename = FLAGS_mask;
  std::vector<cudla_lightnet::Colormap> seg_cmap;
  if (filename != "not-specified") {
    std::vector<std::string> color_list = loadListFromTextFile(filename);    
    for (int i = 0; i < (int)color_list.size(); i++) {
      if (i == 0) {
	//Skip header
	continue;
      }
      std::string colormapString = color_list[i];
      cudla_lightnet::Colormap cmap; 
      std::vector<int> rgb;
      size_t npos = colormapString.find_first_of(',');      
      assert(npos != std::string::npos);
      int id = (int)std::stoi(trim(colormapString.substr(0, npos)));
      colormapString.erase(0, npos + 1);
      
      npos = colormapString.find_first_of(',');
      assert(npos != std::string::npos);      
      std::string name = (trim(colormapString.substr(0, npos)));
      cmap.id = id;
      cmap.name = name;
      colormapString.erase(0, npos + 1);
      while (!colormapString.empty()) {
	size_t npos = colormapString.find_first_of(',');
	if (npos != std::string::npos) {
	  unsigned char c = (unsigned char)std::stoi(trim(colormapString.substr(0, npos)));
	  cmap.color.push_back(c);
	  colormapString.erase(0, npos + 1);
	} else {
	  unsigned char c = (unsigned char)std::stoi(trim(colormapString));
	  cmap.color.push_back(c);
	  break;
	}      
      }
      
      seg_cmap.push_back(cmap);
    }
  }
  return seg_cmap;
}

std::vector<std::string>
get_names(void)
{
  std::string filename = FLAGS_names;
  std::vector<std::string> names;
  if (filename != "not-specified") {
    names = loadListFromTextFile(filename);    
  }
  return names;
}

int
get_num_anchors(void)
{
  int num = FLAGS_num_anchors;
  return num;
}

std::vector<int>
get_anchors(void)
{
  std::string anchorsString = FLAGS_anchors;
  std::vector<int> anchors;
  if (anchorsString != "not-specified") {
    while (!anchorsString.empty()) {
      size_t npos = anchorsString.find_first_of(',');
      if (npos != std::string::npos) {
	int value = (int)std::stoi(trim(anchorsString.substr(0, npos)));
	anchors.push_back(value);
	anchorsString.erase(0, npos + 1);
      } else {
	int value = (int)std::stoi(trim(anchorsString));
	anchors.push_back(value);
	break;
      }      
    }    
  }
  return anchors;
}


double
get_score_thresh(void)
{
  return FLAGS_thresh;
}

  
bool
get_cuda_flg(void)
{
  return FLAGS_cuda;
}

bool
get_sparse_flg(void)
{
  return FLAGS_sparse;
}


bool
get_first_flg(void)
{
  return FLAGS_first;
}

bool
get_last_flg(void)
{
  return FLAGS_last;
}

std::string
get_dump_path(void)
{
  return FLAGS_dump;
}

std::string
get_output_path(void)
{
  return FLAGS_output;
}

double
get_clip_value(void)
{
  return FLAGS_clip;
}

static bool isFlagDefault(std::string flag) { return flag == "not-specified" ? true : false; }

bool getSaveDetections()
{
  if (FLAGS_save_detections)
    assert(!isFlagDefault(FLAGS_save_detections_path)
	   && "save_detections path has to be set if save_detections is set to true");
  return FLAGS_save_detections;
}

std::string getSaveDetectionsPath() { return FLAGS_save_detections_path; }

Window_info
get_window_info(void)
{
  Window_info winfo = {(unsigned int)FLAGS_wx, (unsigned int)FLAGS_wy, (unsigned int)FLAGS_ww, (unsigned int)FLAGS_wh};
  return winfo;
}


//For subnet

std::string
get_subnet_engine_path(void)
{
  return FLAGS_subnet_engine;
}

std::vector<std::string>
get_subnet_names(void)
{
  std::string filename = FLAGS_subnet_names;
  std::vector<std::string> names;
  if (filename != "not-specified") {
    names = loadListFromTextFile(filename);    
  }
  return names;
}

int
get_subnet_num_anchors(void)
{
  int num = FLAGS_subnet_num_anchors;
  return num;
}

std::vector<int>
get_subnet_anchors(void)
{
  std::string anchorsString = FLAGS_subnet_anchors;
  std::vector<int> anchors;
  if (anchorsString != "not-specified") {
    while (!anchorsString.empty()) {
      size_t npos = anchorsString.find_first_of(',');
      if (npos != std::string::npos) {
	int value = (int)std::stoi(trim(anchorsString.substr(0, npos)));
	anchors.push_back(value);
	anchorsString.erase(0, npos + 1);
      } else {
	int value = (int)std::stoi(trim(anchorsString));
	anchors.push_back(value);
	break;
      }      
    }    
  }
  return anchors;
}

int
get_subnet_classes(void)
{
  return FLAGS_subnet_c;
}

std::vector<std::vector<int>>
get_subnet_colormap(void)
{
  std::string filename = FLAGS_subnet_rgb;
  std::vector<std::vector<int>> colormap;
  if (filename != "not-specified") {
    std::vector<std::string> color_list = loadListFromTextFile(filename);    
    for (int i = 0; i < (int)color_list.size(); i++) {
      std::string colormapString = color_list[i];
      std::vector<int> rgb;
      while (!colormapString.empty()) {
	size_t npos = colormapString.find_first_of(',');
	if (npos != std::string::npos) {
	  int colormap = (int)std::stoi(trim(colormapString.substr(0, npos)));
	  rgb.push_back(colormap);
	  colormapString.erase(0, npos + 1);
	} else {
	  int colormap = (int)std::stoi(trim(colormapString));
	  rgb.push_back(colormap);
	  break;
	}      
      }
      colormap.push_back(rgb);
    }
  }
  return colormap;
}

std::vector<std::string>
get_target_names(void)
{
  std::string filename = FLAGS_target_names;
  std::vector<std::string> names;
  if (filename != "not-specified") {
    names = loadListFromTextFile(filename);    
  }
  return names;
}


std::vector<std::string>
get_bluron_names(void)
{
  std::string filename = FLAGS_bluron;
  std::vector<std::string> names;
  if (filename != "not-specified") {
    names = loadListFromTextFile(filename);    
  }
  return names;
}

std::vector<std::string>
get_debug_tensors(void)
{
  std::string debug_tensors_string = FLAGS_debug_tensors;
  std::vector<std::string> debug_tensors;
  if (debug_tensors_string != "not-specified") {
    while (!debug_tensors_string.empty()) {
      size_t npos = debug_tensors_string.find_first_of(',');
      if (npos != std::string::npos) {
	std::string value = trim(debug_tensors_string.substr(0, npos));
	debug_tensors.push_back(value);
	debug_tensors_string.erase(0, npos + 1);
      } else {
	std::string value = trim(debug_tensors_string);
	debug_tensors.push_back(value);
	break;
      }      
    }    
  }
  return debug_tensors;
}


bool
get_save_debug_tensors(void)
{
  return FLAGS_save_debug_tensors;
}
