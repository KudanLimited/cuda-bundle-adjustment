########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(g2o_COMPONENT_NAMES ${g2o_COMPONENT_NAMES} )
list(REMOVE_DUPLICATES g2o_COMPONENT_NAMES)
set(g2o_FIND_DEPENDENCY_NAMES ${g2o_FIND_DEPENDENCY_NAMES} Eigen3)
list(REMOVE_DUPLICATES g2o_FIND_DEPENDENCY_NAMES)

########### VARIABLES #######################################################################
#############################################################################################
set(g2o_PACKAGE_FOLDER_RELWITHDEBINFO "/home/kudan/.conan/data/g2o/20201223/_/_/package/5e481ef36063949e2b86820fabbf7b010b748072")
set(g2o_INCLUDE_DIRS_RELWITHDEBINFO "${g2o_PACKAGE_FOLDER_RELWITHDEBINFO}/include")
set(g2o_RES_DIRS_RELWITHDEBINFO "${g2o_PACKAGE_FOLDER_RELWITHDEBINFO}/res")
set(g2o_DEFINITIONS_RELWITHDEBINFO )
set(g2o_SHARED_LINK_FLAGS_RELWITHDEBINFO )
set(g2o_EXE_LINK_FLAGS_RELWITHDEBINFO )
set(g2o_OBJECTS_RELWITHDEBINFO )
set(g2o_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(g2o_COMPILE_OPTIONS_C_RELWITHDEBINFO )
set(g2o_COMPILE_OPTIONS_CXX_RELWITHDEBINFO )
set(g2o_LIB_DIRS_RELWITHDEBINFO "${g2o_PACKAGE_FOLDER_RELWITHDEBINFO}/lib")
set(g2o_LIBS_RELWITHDEBINFO g2o_core g2o_solver_dense g2o_solver_eigen g2o_solver_pcg g2o_solver_structure_only g2o_stuff g2o_types_data g2o_types_icp g2o_types_sba g2o_types_sclam2d g2o_types_sim3 g2o_types_slam2d g2o_types_slam2d_addons g2o_types_slam3d g2o_types_slam3d_addons)
set(g2o_SYSTEM_LIBS_RELWITHDEBINFO )
set(g2o_FRAMEWORK_DIRS_RELWITHDEBINFO "${g2o_PACKAGE_FOLDER_RELWITHDEBINFO}/Frameworks")
set(g2o_FRAMEWORKS_RELWITHDEBINFO )
set(g2o_BUILD_MODULES_PATHS_RELWITHDEBINFO )
set(g2o_BUILD_DIRS_RELWITHDEBINFO "${g2o_PACKAGE_FOLDER_RELWITHDEBINFO}/")

set(g2o_COMPONENTS_RELWITHDEBINFO )