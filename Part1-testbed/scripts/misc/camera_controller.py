#!/usr/bin/env python3
"""
CARLA camera control script with keyboard input.
Controls camera position and orientation using keyboard.
Displays xyz coordinates and theta angles on screen and prints to stdout.
"""

import carla
import pygame
import cv2
import numpy as np
import os
import sys
import signal
from queue import Queue, Empty

# Prefer scripts/util.py over CARLA PythonAPI/util
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import util

# Movement parameters
POSITION_STEP = 2.5  # meters per key press
ROTATION_STEP = 2.0  # degrees per key press

# Initial camera position (from util.py default)
INITIAL_LOCATION = carla.Location(x=151.105438, y=-200.910126, z=8.275307)
INITIAL_ROTATION = carla.Rotation(pitch=-15.0, yaw=-178.560471, roll=0.0)

# Camera save file
CAMERA_SAVE_FILE = "camera_configs.txt"

def cleanup_camera(camera):
    """Clean up the camera."""
    if camera is not None:
        print("Stopping camera...")
        camera.stop()
        camera.destroy()
    print("Cleanup complete.")

def get_euler_angles(rotation):
    """Convert CARLA rotation to Euler angles (pitch, yaw, roll) in degrees."""
    return rotation.pitch, rotation.yaw, rotation.roll

def print_camera_info(location, rotation):
    """Print camera position and orientation to stdout."""
    pitch, yaw, roll = get_euler_angles(rotation)
    print(f"Position: x={location.x:.3f}, y={location.y:.3f}, z={location.z:.3f} | "
          f"Rotation: pitch={pitch:.2f}°, yaw={yaw:.2f}°, roll={roll:.2f}°")

def save_camera_config(camera_id, location, rotation):
    """Save camera configuration to file in util.py format."""
    pitch, yaw, roll = get_euler_angles(rotation)
    
    # Format: {"id": X, "pos": (x, y, z), "rot": (pitch, yaw, roll)}
    config_line = f'    {{"id": {camera_id}, "pos": ({location.x:.3f}, {location.y:.3f}, {location.z:.3f}), "rot": ({pitch:.2f}, {yaw:.2f}, {roll:.2f})}},\n'
    
    # Append to file
    with open(CAMERA_SAVE_FILE, "a") as f:
        f.write(config_line)
    
    print(f"\n✓ Camera {camera_id} saved to {CAMERA_SAVE_FILE}")
    print(f"  Position: ({location.x:.3f}, {location.y:.3f}, {location.z:.3f})")
    print(f"  Rotation: ({pitch:.2f}, {yaw:.2f}, {roll:.2f})")

def draw_info_overlay(image, location, rotation):
    """Draw camera position and orientation on the image."""
    pitch, yaw, roll = get_euler_angles(rotation)
    
    # Create overlay text
    info_lines = [
        f"Position:",
        f"  X: {location.x:.3f}",
        f"  Y: {location.y:.3f}",
        f"  Z: {location.z:.3f}",
        f"",
        f"Rotation:",
        f"  Pitch: {pitch:.2f}°",
        f"  Yaw: {yaw:.2f}°",
        f"  Roll: {roll:.2f}°",
        f"",
        f"Controls:",
        f"  WASD: Move X/Y",
        f"  Q/E: Move Z (up/down)",
        f"  Arrow Keys: Rotate pitch/yaw",
        f"  U/O: Rotate roll",
        f"  F: Save camera config",
        f"  ESC: Exit"
    ]
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (400, 350), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Draw text
    y_offset = 30
    for i, line in enumerate(info_lines):
        if line.startswith("Position:") or line.startswith("Rotation:") or line.startswith("Controls:"):
            cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif line.startswith("  "):
            cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    return image

def main():
    # Initialize pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("CARLA Camera Control")
    clock = pygame.time.Clock()
    
    # Connect to CARLA
    print("Connecting to CARLA at localhost:2000...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        print(f"Connected to CARLA. Map: {world.get_map().name}")
    except Exception as e:
        print(f"ERROR: Cannot connect to CARLA server: {e}")
        pygame.quit()
        sys.exit(1)
    
    # Setup camera
    cam_bp, cam_tf = util.create_camera(world)
    camera = world.try_spawn_actor(cam_bp, cam_tf)
    
    if camera is None:
        print("ERROR: Failed to spawn camera.")
        pygame.quit()
        sys.exit(1)
    
    print(f"Camera spawned successfully. ID: {camera.id}")
    
    # Setup camera frame capture
    q = Queue()
    camera.listen(q.put)
    
    # Setup cleanup handler
    def signal_handler(sig, frame):
        cleanup_camera(camera)
        pygame.quit()
        cv2.destroyAllWindows()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Current camera transform
    current_transform = camera.get_transform()
    
    print("\nControls:")
    print("  W/S:        Move forward/backward (Y axis)")
    print("  A/D:        Move left/right (X axis)")
    print("  Q/E:        Move up/down (Z axis)")
    print("  Arrow Up/Down:   Rotate pitch")
    print("  Arrow Left/Right: Rotate yaw")
    print("  U/O:        Rotate roll")
    print("  F:          Save camera config")
    print("  ESC:        Exit")
    print(f"\nCamera configs will be saved to: {CAMERA_SAVE_FILE}")
    print("Ready! Use keyboard to control the camera.")
    print_camera_info(current_transform.location, current_transform.rotation)
    
    # Camera ID counter (starts at 1)
    camera_id = 1
    
    # Main loop
    running = True
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    # Save current camera configuration
                    save_camera_config(camera_id, current_transform.location, current_transform.rotation)
                    camera_id += 1
        
        # Get current key states
        keys = pygame.key.get_pressed()
        transform_changed = False
        
        # Position controls
        new_location = carla.Location(
            x=current_transform.location.x,
            y=current_transform.location.y,
            z=current_transform.location.z
        )
        if keys[pygame.K_d]:  # Forward (Y+)
            new_location.y += POSITION_STEP
            transform_changed = True
        if keys[pygame.K_a]:  # Backward (Y-)
            new_location.y -= POSITION_STEP
            transform_changed = True
        if keys[pygame.K_s]:  # Left (X-)
            new_location.x -= POSITION_STEP
            transform_changed = True
        if keys[pygame.K_w]:  # Right (X+)
            new_location.x += POSITION_STEP
            transform_changed = True
        if keys[pygame.K_q]:  # Up (Z+)
            new_location.z += POSITION_STEP
            transform_changed = True
        if keys[pygame.K_e]:  # Down (Z-)
            new_location.z -= POSITION_STEP
            transform_changed = True
        
        # Rotation controls
        new_rotation = carla.Rotation(
            pitch=current_transform.rotation.pitch,
            yaw=current_transform.rotation.yaw,
            roll=current_transform.rotation.roll
        )
        if keys[pygame.K_UP]:  # Pitch up
            new_rotation.pitch = min(90.0, new_rotation.pitch + ROTATION_STEP)
            transform_changed = True
        if keys[pygame.K_DOWN]:  # Pitch down
            new_rotation.pitch = max(-90.0, new_rotation.pitch - ROTATION_STEP)
            transform_changed = True
        if keys[pygame.K_LEFT]:  # Yaw left
            new_rotation.yaw = (new_rotation.yaw - ROTATION_STEP) % 360.0
            transform_changed = True
        if keys[pygame.K_RIGHT]:  # Yaw right
            new_rotation.yaw = (new_rotation.yaw + ROTATION_STEP) % 360.0
            transform_changed = True
        if keys[pygame.K_u]:  # Roll left
            new_rotation.roll = (new_rotation.roll - ROTATION_STEP) % 360.0
            transform_changed = True
        if keys[pygame.K_o]:  # Roll right
            new_rotation.roll = (new_rotation.roll + ROTATION_STEP) % 360.0
            transform_changed = True
        
        # Update camera transform if changed
        if transform_changed:
            current_transform = carla.Transform(new_location, new_rotation)
            camera.set_transform(current_transform)
            
            # Print to stdout only when transform changes
            print_camera_info(current_transform.location, current_transform.rotation)
        
        # Get camera frame
        try:
            frame = q.get(timeout=0.1)
            arr = np.frombuffer(frame.raw_data, np.uint8).reshape(
                (frame.height, frame.width, 4))[:, :, :3].copy()
            
            # Draw info overlay
            arr = draw_info_overlay(arr, current_transform.location, current_transform.rotation)
            
            # Display image
            cv2.imshow("CARLA Camera Control", arr)
            
            # Exit on ESC (OpenCV window)
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
        
        except Empty:
            pass
        
        # Update pygame display (keep window responsive)
        pygame.display.flip()
        clock.tick(30)  # 30 FPS for responsive control
    
    # Cleanup
    cleanup_camera(camera)
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
