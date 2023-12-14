#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <sys/stat.h>
#include <iostream>

#include "cudaRenderer.h"
#include "refRenderer.h"
#include "platformgl.h"

bool createDirectory(const std::string& path) {
    struct stat info;

    if (stat(path.c_str(), &info) != 0) {
        // Directory does not exist, try to create it
        if (mkdir(path.c_str(), 0777) == -1) {
            std::cerr << "Error: Unable to create directory " << path << std::endl;
            return false;
        }
    }
    return true;
}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -b  --baseline  Run the sequential baseline simulation\n");
    printf("  -h  --help                 This message\n");
}

void startRendererWithDisplay(CircleRenderer* renderer); // From display.cpp

int main(int argc, char** argv)
{
    createDirectory("output_images");
    int imageWidth = 512;
    int imageHeight = 512;
    bool useBaseline = false;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"help",     0, 0,  'h'},
        {"baseline", 0, 0,  'b'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "hb", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'b':
            useBaseline = true;
            printf("Running baseline simulation...\n");
            break;
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    printf("Rendering %dx%d simulation\n", imageWidth, imageHeight);

    CircleRenderer* renderer;
   
    renderer = new CudaRenderer();

    renderer->allocOutputImage(imageWidth, imageHeight);
    renderer->setup();

    glutInit(&argc, argv);
    startRendererWithDisplay(renderer);

    return 0;
}
