#pragma once
#include "vk_common.h"

struct GLFWwindow;

namespace somegi {

struct WindowDesc {
    int width = 1600;
    int height = 900;
    const char* title = "SomeGI";
};

class Window {
public:
    explicit Window(const WindowDesc& d);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const;
    void pollEvents();

    GLFWwindow* handle() const { return m_window; }
    int width() const { return m_w; }
    int height() const { return m_h; }
    bool resized();
    void onResize(int w, int h);

    VkSurfaceKHR createSurface(VkInstance instance);

private:
    GLFWwindow* m_window = nullptr;
    int m_w = 0, m_h = 0;
    bool m_resized = false;
};

}
