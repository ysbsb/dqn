import setup_path
import airsim

quad_state = client.getMultirotorState().kinematics_estimated.position
print(quad_state)
quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
print(quad_vel)
collision_info = client.simGetCollisionInfo()
print(collision_info0)
