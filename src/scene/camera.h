#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct GLFWwindow;

namespace somegi {

class Camera {
public:
    glm::vec3 position{0.0f, 1.0f, 3.0f};
    float yaw = -90.0f;
    float pitch = 0.0f;
    float fovDeg = 60.0f;
    float nearZ = 0.5f;
    float farZ = 200.0f;

    glm::vec3 forward() const;
    glm::vec3 right() const;
    glm::vec3 up() const;

    glm::mat4 view() const;
    glm::mat4 proj(float aspect) const;
};

class FlyController {
public:
    void update(Camera& cam, float dtSec, GLFWwindow* window);

    // Base WASD/QE speed (units/sec). Shift multiplies by 4×.
    // Set per scene so movement scales with world size.
    float moveSpeed = 3.0f;
private:
    bool m_dragging = false;
    double m_lastX = 0, m_lastY = 0;
};

}
