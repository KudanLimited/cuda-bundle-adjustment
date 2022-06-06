from conans import ConanFile, tools
from conan.tools.cmake import CMakeDeps, CMakeToolchain, CMake
from pathlib import Path
import os


class CugoConan(ConanFile):
    name = "cuba"
    license = "Apache License 2.0"
    author = "fixstars"
    url = "https://github.com/fixstars/cuda-bundle-adjustment"
    description = "A CUDA implementation of Bundle Adjustment"
    topics = ("cuda", "ba", "bundle adjustment")
    
    settings = "os", "compiler", "build_type", "arch"
    exports = ["*.tar.gz", "CMakeLists.txt"]
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_samples": [True, False],
        "with_g2o": [True, False],
        "use_float32": [True, False]
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        "with_samples": True,
        "with_g2o": False,
        "use_float32": False
    }
    _cmake = None

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
    
    def requirements(self):
        self.requires("opencv/4.0.1")
        self.requires("eigen/3.3.9")
        if self.options.with_g2o:
            self.requires("g2o/20201223")
       
    def layout(self):
        self.folders.source = "./src"
        self.folders.build = os.getcwd()
        self.folders.generators = f"{self.folders.build}/Conan"
         
    def generate(self):
        tc = CMakeToolchain(self)

        tc.variables["ENABLE_SAMPLES"] = self.options.with_samples
        tc.variables["WITH_G2O"] = self.options.with_g2o
        tc.variables["USE_FLOAT32"] = self.options.use_float32
        tc.variables["BUILD_CUGO_SHARED"] = bool(self.options.shared)
        
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()
        
    def build(self):
        self._cmake = CMake(self)
        self._cmake.configure()
        self._cmake.build()

    def package(self):
        self.copy("LICENSE", dst="licenses")

        if self._cmake is None:
            self._cmake = CMake(self)
            self._cmake.configure()
        self._cmake.install()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        self.cpp_info.set_property("cmake_file_name", "cugo")
        self.cpp_info.set_property("cmake_target_name", "cugo::cugo")

