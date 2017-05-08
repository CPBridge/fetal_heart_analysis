cmake_minimum_required(VERSION 3.5)
project(fetal_heart_analysis)

# Make sure we're using C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Main executables
add_executable(train_rotinv exec/src/train_rotinv.cpp)
add_executable(test_rotinv exec/src/test_rotinv.cpp)
add_executable(train_square exec/src/train_square.cpp)
add_executable(test_square exec/src/test_square.cpp)

# Some g++specific optimisations (feel free to add equivalents for other
# compilers)
if(CMAKE_COMPILER_IS_GNUCXX)
	set(GCC_OPTIMISATION_FLAGS "-ffast-math -DEIGEN_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GCC_OPTIMISATION_FLAGS}" )
endif()

# Third party dependencies
# ------------------------

# Dependency on Boost libraries
find_package( Boost REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )
target_link_libraries( train_rotinv ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} )
target_link_libraries( train_square ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} )
target_link_libraries( test_rotinv ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} )
target_link_libraries( test_square ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} )

# Dependency on OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( train_rotinv ${OpenCV_LIBS} )
target_link_libraries( train_square ${OpenCV_LIBS} )
target_link_libraries( test_rotinv  ${OpenCV_LIBS} )
target_link_libraries( test_square  ${OpenCV_LIBS} )

# Dependency on OpenMP
find_package(OpenMP REQUIRED)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Dependency on eigen3
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

# External dependencies (code in other repos)
# -------------------------------------------

# Dependency on the "canopy" repository
set(CANOPY_DIR "" CACHE PATH "Root directory of the canopy repository")
include_directories(${CANOPY_DIR}/include)

# Dependency on the "other" directory
set(RIFEATURES_DIR "" CACHE PATH "Root directory of the RIFeatures repository")
include_directories(${RIFEATURES_DIR}/include)
add_subdirectory(${RIFEATURES_DIR} RIFeatures)
target_link_libraries(train_rotinv RIFeatures)
target_link_libraries(test_rotinv  RIFeatures)

# Dependency on the "monogenic" repository
set(MONOGENIC_DIR "" CACHE PATH "Root directory of the monogenic repository")
include_directories(${MONOGENIC_DIR}/include)
add_subdirectory(${MONOGENIC_DIR} monogenic)
target_link_libraries(train_rotinv monogenic)
target_link_libraries(train_square monogenic)
target_link_libraries(test_rotinv  monogenic)
target_link_libraries(test_square  monogenic)

# Internal dependencies
# ---------------------

# Dependency on the "utilities" directory
set(UTILITIES_DIR "utilities/")
include_directories(${UTILITIES_DIR}/include)
add_subdirectory(${UTILITIES_DIR})
target_link_libraries(train_rotinv utilities)
target_link_libraries(train_square utilities)
target_link_libraries(test_rotinv  utilities)
target_link_libraries(test_square  utilities)

# Dependency on the "features" directory
set(FEATURES_DIR "features/")
include_directories(${FEATURES_DIR}/include)
add_subdirectory(${FEATURES_DIR})
target_link_libraries(train_rotinv features)
target_link_libraries(train_square features)
target_link_libraries(test_rotinv  features)
target_link_libraries(test_square  features)

# Dependency on other header-only directories
include_directories(forests/include)
include_directories(filters/include)
