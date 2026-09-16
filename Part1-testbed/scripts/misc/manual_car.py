#!/usr/bin/env python3
"""
Manual car control script using pygame for keyboard input.
Controls a CARLA vehicle with arrow keys.
Supports multiple simultaneous key presses (e.g., up + right for turning right).
"""

import pygame
import carla
import sys
import signal

# Hardcoded spawn location (from util.py)
SPAWN_LOCATION = carla.Location(x=151.105438, y=-200.910126, z=8.275307)
SPAWN_ROTATION = carla.Rotation(pitch=-15.000000, yaw=-178.560471, roll=0.000000)

# Control parameters
THROTTLE_VALUE = 1  # Forward throttle when up arrow is pressed
REVERSE_VALUE = 0.3   # Reverse throttle when down arrow is pressed
BRAKE_VALUE = 1     # Brake when no movement keys are pressed
STEER_STEP = 0.05     # Steering increment per frame
MAX_STEER = 0.7       # Maximum steering angle
STEER_DECAY = 0.02    # Steering return-to-center rate

def cleanup_vehicle(client, vehicle):
    """Clean up the spawned vehicle."""
    if vehicle is not None:
        print("Destroying vehicle...")
        vehicle.destroy()
    print("Cleanup complete.")

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("CARLA Manual Control")
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
    
    # Check that CARLA is in async mode
    settings = world.get_settings()
    if settings.synchronous_mode:
        print("WARNING: CARLA is in synchronous mode. This script assumes async mode.")
    
    # Spawn vehicle
    print("Spawning vehicle...")
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]  # Use Tesla Model 3 as default
    
    spawn_transform = carla.Transform(SPAWN_LOCATION, SPAWN_ROTATION)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    
    if vehicle is None:
        print("ERROR: Failed to spawn vehicle. Position may be occupied.")
        pygame.quit()
        sys.exit(1)
    
    print(f"Vehicle spawned successfully. ID: {vehicle.id}")

    world.get_spectator().set_transform(spawn_transform)
    
    # Setup cleanup handler
    def signal_handler(sig, frame):
        cleanup_vehicle(client, vehicle)
        pygame.quit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Control state
    current_steer = 0.0
    
    print("\nControls:")
    print("  Arrow Up:    Forward")
    print("  Arrow Down:  Reverse")
    print("  Arrow Left:  Steer Left")
    print("  Arrow Right: Steer Right")
    print("  (Can combine keys, e.g., Up+Right for right turn)")
    print("  ESC:         Exit")
    print("\nReady! Use arrow keys to control the vehicle.")
    
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
        
        # Get current key states (allows multiple keys)
        keys = pygame.key.get_pressed()
        
        # Initialize control
        control = carla.VehicleControl()
        
        # Throttle (Up arrow) - Forward
        if keys[pygame.K_UP]:
            control.throttle = THROTTLE_VALUE
            control.reverse = False
            control.brake = 0.0
        
        # Reverse (Down arrow)
        elif keys[pygame.K_DOWN]:
            control.throttle = REVERSE_VALUE
            control.reverse = True
            control.brake = 0.0
        
        # No movement keys - apply brake
        else:
            control.throttle = 0.0
            control.brake = BRAKE_VALUE
            control.reverse = False
        
        # Steering (Left/Right arrows) - can be combined with movement
        if keys[pygame.K_LEFT]:
            current_steer = max(-MAX_STEER, current_steer - STEER_STEP)
        elif keys[pygame.K_RIGHT]:
            current_steer = min(MAX_STEER, current_steer + STEER_STEP)
        else:
            # Gradually return steering to center when no steering key is pressed
            if abs(current_steer) < STEER_DECAY:
                current_steer = 0.0
            elif current_steer > 0:
                current_steer -= STEER_DECAY
            else:
                current_steer += STEER_DECAY
        
        control.steer = current_steer
        
        # Apply control to vehicle
        vehicle.apply_control(control)
        
        # Update display (keep window responsive)
        pygame.display.flip()
        clock.tick(60)  # 60 FPS for responsive control
    
    # Cleanup
    cleanup_vehicle(client, vehicle)
    pygame.quit()

if __name__ == "__main__":
    main()
