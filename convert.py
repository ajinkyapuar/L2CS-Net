import math

def convert_gaze_to_screen(pitch, yaw, head_position, screen_distance):
    # Define the screen properties (you may need to adjust these values)
    screen_width = 480  # Screen width in pixels
    screen_height = 640  # Screen height in pixels

    # Define the head position (assuming head is centered at (0, 0, 0))
    head_x, head_y, head_z = head_position

    # Convert pitch and yaw angles to radians
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    # Calculate gaze direction vector
    x = math.sin(yaw_rad) * math.cos(pitch_rad)
    y = math.sin(pitch_rad)
    z = math.cos(yaw_rad) * math.cos(pitch_rad)

    # Calculate gaze vector in world coordinates
    gaze_vector = (x, y, z)

    # Calculate intersection point with screen plane
    screen_x = (head_x + screen_distance * gaze_vector[0]) / gaze_vector[2]
    screen_y = (head_y + screen_distance * gaze_vector[1]) / gaze_vector[2]

    # Convert screen coordinates to pixel coordinates
    pixel_x = int((screen_x + 0.5) * screen_width)
    pixel_y = int((screen_y + 0.5) * screen_height)

    # Return the screen gaze point as pixel coordinates
    return pixel_x, pixel_y
