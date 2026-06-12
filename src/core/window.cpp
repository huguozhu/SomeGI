#include "window.h"
#include <GLFW/glfw3.h>
#include <stdexcept>

namespace somegi {

static void framebufferResizeCb(GLFWwindow* w, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self) self->onResize(width, height);
}

int Window::s_liveCount = 0;

Window::Window(const WindowDesc& d) : m_w(d.width), m_h(d.height) {
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    ++s_liveCount;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(d.width, d.height, d.title, nullptr, nullptr);
    if (!m_window) throw std::runtime_error("glfwCreateWindow failed");
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCb);
}

Window::~Window() {
    if (m_window) glfwDestroyWindow(m_window);
    // glfwTerminate() 只在最后一个窗口销毁时调用，避免多窗口场景下提前终止 GLFW
    if (--s_liveCount == 0) glfwTerminate();
}

bool Window::shouldClose() const { return glfwWindowShouldClose(m_window); }
void Window::pollEvents() { glfwPollEvents(); }
bool Window::resized() { bool r = m_resized; m_resized = false; return r; }
void Window::onResize(int w, int h) { m_w = w; m_h = h; m_resized = true; }

VkSurfaceKHR Window::createSurface(VkInstance instance) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult r = glfwCreateWindowSurface(instance, m_window, nullptr, &surface);
    if (r != VK_SUCCESS) throw std::runtime_error("glfwCreateWindowSurface failed");
    return surface;
}

}
