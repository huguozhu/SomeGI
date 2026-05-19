#include "camera.h"
#include <GLFW/glfw3.h>

namespace somegi {

glm::vec3 Camera::forward() const {
    float y = glm::radians(yaw), p = glm::radians(pitch);
    return glm::normalize(glm::vec3(cos(y)*cos(p), sin(p), sin(y)*cos(p)));
}
glm::vec3 Camera::right() const {
    return glm::normalize(glm::cross(forward(), glm::vec3(0,1,0)));
}
glm::vec3 Camera::up() const {
    return glm::normalize(glm::cross(right(), forward()));
}
glm::mat4 Camera::view() const {
    return glm::lookAt(position, position + forward(), glm::vec3(0,1,0));
}
glm::mat4 Camera::proj(float aspect) const {
    auto p = glm::perspective(glm::radians(fovDeg), aspect, nearZ, farZ);
    p[1][1] *= -1.0f;
    return p;
}

void FlyController::update(Camera& cam, float dt, GLFWwindow* w) {
    int rmb = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT);
    double mx, my; glfwGetCursorPos(w, &mx, &my);
    if (rmb == GLFW_PRESS) {
        if (!m_dragging) { m_dragging = true; m_lastX = mx; m_lastY = my; }
        float dx = float(mx - m_lastX), dy = float(my - m_lastY);
        m_lastX = mx; m_lastY = my;
        cam.yaw   += dx * 0.15f;
        cam.pitch -= dy * 0.15f;
        cam.pitch = glm::clamp(cam.pitch, -89.0f, 89.0f);
    } else {
        m_dragging = false;
    }

    float speed = moveSpeed;
    if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 4.0f;
    glm::vec3 d{0};
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) d += cam.forward();
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) d -= cam.forward();
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) d -= cam.right();
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) d += cam.right();
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) d -= glm::vec3(0,1,0);
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) d += glm::vec3(0,1,0);
    if (glm::length(d) > 0) cam.position += glm::normalize(d) * speed * dt;
}

}
