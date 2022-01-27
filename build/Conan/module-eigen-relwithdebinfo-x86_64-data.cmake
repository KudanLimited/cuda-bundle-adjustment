########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(eigen_COMPONENT_NAMES ${eigen_COMPONENT_NAMES} Eigen3::eigen3)
list(REMOVE_DUPLICATES eigen_COMPONENT_NAMES)
set(eigen_FIND_DEPENDENCY_NAMES ${eigen_FIND_DEPENDENCY_NAMES} )
list(REMOVE_DUPLICATES eigen_FIND_DEPENDENCY_NAMES)

########### VARIABLES #######################################################################
#############################################################################################
set(eigen_PACKAGE_FOLDER_RELWITHDEBINFO "/home/kudan/.conan/data/eigen/3.3.9/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9")
set(eigen_INCLUDE_DIRS_RELWITHDEBINFO "${eigen_PACKAGE_FOLDER_RELWITHDEBINFO}/include/eigen3")
set(eigen_RES_DIRS_RELWITHDEBINFO )
set(eigen_DEFINITIONS_RELWITHDEBINFO )
set(eigen_SHARED_LINK_FLAGS_RELWITHDEBINFO )
set(eigen_EXE_LINK_FLAGS_RELWITHDEBINFO )
set(eigen_OBJECTS_RELWITHDEBINFO )
set(eigen_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(eigen_COMPILE_OPTIONS_C_RELWITHDEBINFO )
set(eigen_COMPILE_OPTIONS_CXX_RELWITHDEBINFO )
set(eigen_LIB_DIRS_RELWITHDEBINFO )
set(eigen_LIBS_RELWITHDEBINFO )
set(eigen_SYSTEM_LIBS_RELWITHDEBINFO m)
set(eigen_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(eigen_FRAMEWORKS_RELWITHDEBINFO )
set(eigen_BUILD_MODULES_PATHS_RELWITHDEBINFO )
set(eigen_BUILD_DIRS_RELWITHDEBINFO )

set(eigen_COMPONENTS_RELWITHDEBINFO Eigen3::eigen3)
########### COMPONENT Eigen3::eigen3 VARIABLES #############################################
set(eigen_Eigen3_eigen3_INCLUDE_DIRS_RELWITHDEBINFO "${eigen_PACKAGE_FOLDER_RELWITHDEBINFO}/include/eigen3")
set(eigen_Eigen3_eigen3_LIB_DIRS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_RES_DIRS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_DEFINITIONS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_OBJECTS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_COMPILE_DEFINITIONS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_COMPILE_OPTIONS_C_RELWITHDEBINFO "")
set(eigen_Eigen3_eigen3_COMPILE_OPTIONS_CXX_RELWITHDEBINFO "")
set(eigen_Eigen3_eigen3_LIBS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_SYSTEM_LIBS_RELWITHDEBINFO m)
set(eigen_Eigen3_eigen3_FRAMEWORK_DIRS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_FRAMEWORKS_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_DEPENDENCIES_RELWITHDEBINFO )
set(eigen_Eigen3_eigen3_LINKER_FLAGS_RELWITHDEBINFO
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)