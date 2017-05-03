add_library(features src/histograms.cpp src/imageFeatureProcessor.cpp src/squareTrainingFunctors.cpp src/squareTestingFunctors.cpp src/orientationTestingFunctor.cpp src/trainingFunctors.cpp) 
target_link_libraries( features ${OpenCV_LIBS} )
target_link_libraries(features monogenic)
