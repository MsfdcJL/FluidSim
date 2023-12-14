#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <iostream>

#include "circleRenderer.h"
#include "cycleTimer.h"
#include "image.h"
#include "platformgl.h"

class FluidSimulator {
public:
    FluidSimulator(int width, int height)
        : width(width), height(height), printStats(true), lastFrameTime(0.0) {
        renderer = new CircleRenderer();
    }

    ~FluidSimulator() {
        delete renderer;
    }

    void handleReshape(int w, int h) {
        width = w;
        height = h;
        glViewport(0, 0, width, height);
    }

    void handleDisplay() {
        if (number == 100 || number == 150) {
            std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
        } else {
            std::cout << "NUMBER " << number << " ";
        }
        number++;

        renderPicture();
        double currentTime = CycleTimer::currentSeconds();
        if (printStats) {
            std::cout << (currentTime - lastFrameTime) * 1000.0 << " ms\n";
        }
        lastFrameTime = currentTime;
    }

    void handleMouseClick(int button, int state, int x, int y) {
        if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
            mousePressedLocations.emplace_back(x, height - y);
        }
    }

    void handleMouseMove(int x, int y) {
        int index = (height - y - 1) * width + x;
        if (0 <= index && index < width * height) {
            mousePressedLocations.emplace_back(x, height - y);
        }
    }

    void startRendering() {
        glutInitWindowSize(width, height);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutCreateWindow("Fluid Simulator");
        glutDisplayFunc([]() { instance->handleDisplay(); });
        glutReshapeFunc([](int w, int h) { instance->handleReshape(w, h); });
        glutMouseFunc([](int button, int state, int x, int y) { instance->handleMouseClick(button, state, x, y); });
        glutMotionFunc([](int x, int y) { instance->handleMouseMove(x, y); });
        glutMainLoop();
    }

    static FluidSimulator* getInstance() {
        if (!instance) {
            instance = new FluidSimulator(512, 512); // Default size
        }
        return instance;
    }

private:
    void renderPicture() {
        double startTime = CycleTimer::currentSeconds();
        renderer->clearImage();
        renderer->setNewQuantities(mousePressedLocations);
        mousePressedLocations.clear();
        renderer->render();
        double endRenderTime = CycleTimer::currentSeconds();
        if (printStats) {
            std::cout << "Clear: " << (startTime - lastFrameTime) * 1000.0 << " ms\n";
            std::cout << "Render: " << (endRenderTime - startTime) * 1000.0 << " ms\n";
        }
    }

    int width, height;
    bool printStats;
    double lastFrameTime;
    std::vector<std::pair<int, int>> mousePressedLocations;
    CircleRenderer* renderer;
    static FluidSimulator* instance;
    int number = 0;
};

FluidSimulator* FluidSimulator::instance = nullptr;

// Usage example (in some main function or other context):
// FluidSimulator::getInstance()->startRendering();
