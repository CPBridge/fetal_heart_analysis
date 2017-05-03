add_library(utilities src/displayFunctions.cpp src/thesisUtilities.cpp)
target_link_libraries( utilities ${OpenCV_LIBS} )
