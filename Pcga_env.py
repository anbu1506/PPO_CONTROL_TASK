import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class PCGAEnv(gym.Env):
    """
    PCGA (Pendulum-Controlled Gravity Apparatus) environment for OpenAI Gym.
    
    The goal is to control the angle of a rotating stick to position a ball at a target distance.
    
    Observation Space:
        - Ball distance from start
        - Target position
    
    Action Space:
        - 0: Increase angle
        - 1: Decrease angle
        - 2: Do nothing
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super(PCGAEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Increase angle, 1: Decrease angle, 2: Do nothing
        
        # Observation space: [ball_distance_from_start, target_position]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -100,84]),  # Min radius is 0, min target position is 0 , minimum velocity is -200 , min angle is 84
            high=np.array([200, 200,100,96]),  # Max radius is 200, max target position is 200, max velocity is 100 , max angle is 96
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.render_flag = render_mode is not None
        self.pcga = PCGA(render=self.render_flag)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate a random target position between 20 and 180
        self.target_position = self.np_random.uniform(20, 120)
        self.pcga.set_target_position(self.target_position)
        
        observation, _, _ = self.pcga.reset()
        info = {}
        
        return np.array(observation, dtype=np.float32), info
    
    def step(self, action):
        # Map action to PCGA action
        pcga_action = action if action < 2 else None
        
        observation, reward, _ = self.pcga.step(pcga_action)
        
        # Convert to proper gym types
        observation = np.array(observation, dtype=np.float32)
        
        # Calculate distance to target
        distance_to_target = abs(observation[0] - observation[1])
        
        # Determine if episode should terminate
        done = False
        
        # End episode on failure: ball falls off
        if observation[0] < 0  or observation[0] > self.pcga.RADIUS1 :
            done = True
            reward -= 100
        
        truncated = self.pcga.time_passed > 30  # 30 seconds time limit
        
        info = {
            "ball_position": observation[0],
            "target_position": observation[1],
            "distance_to_target": distance_to_target
        }
        
        return observation, reward, done, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            raise NotImplementedError("RGB array rendering not implemented")
    
    def close(self):
        if self.render_flag:
            pygame.quit()


class PCGA():
    """Original PCGA implementation"""

    def __init__(self, render=False):
        self.render = render
        if render:
          pygame.init()
          self.font = pygame.font.SysFont('Arial', 16)

        self.width = 400
        self.height = 400

        self.g = 9.8
        self.dt = 1/60

        self.VERTICAL_STICK_1_POSITION = (50, 40)
        self.HEIGHT1 = 100
        self.ANGLE1 = 0
        self.RADIUS1 = 200

        self.VERTICAL_STICK_2_POSITION = (150, 40)
        self.HEIGHT2 = 20
        self.ANGLE2 = 0
        self.RADIUS2 = 100

        self.BALL_RADIUS = 10
        self.I = (2/5)
        
        if render:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        
        self.is_first_time = True
        self.prev_angle = 0
        self.ball_velocity = 0
        self.ball_position = None
        self.ball_distance_from_start = 0
        
        self.time_passed = 0
        
        self.actual_distance = self.HEIGHT1 - self.HEIGHT2
        
        self.MAX_REWARD = self.RADIUS1
        self.target_position = 100  # Default target
        self.eps_reward = 0
        
    def point(self, x, y):
        return (x, self.height - y)
    
    def draw_vertical_stick(self, coor, height):
        base = self.point(coor[0], coor[1])
        top = self.point(coor[0], coor[1] + height)
        if self.render:
            pygame.draw.line(self.screen, (0, 0, 0), base, top, 5)
        return top
    
    def draw_rotating_stick(self, base, length, angle):
        x = base[0] + length * math.cos(angle)
        y = base[1] + length * math.sin(angle)
        if self.render:
            pygame.draw.line(self.screen, (255, 0, 0), base, (x, y), 5)
        return (x, y)
        
    def calculate_angular_displacement(self, base, length, angle):
        x = base[0] + length * math.cos(angle)
        y = base[1] + length * math.sin(angle)
        return (x, y)

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def draw_connector(self, coor1, coor2):
        if self.render:
            pygame.draw.line(self.screen, (0, 255, 0), coor1, coor2, 3)
        
    def draw_ball(self, coor):
        if self.render:
            x = coor[0]
            y = coor[1]
            
            x += self.BALL_RADIUS
            y -= self.BALL_RADIUS
            
            pygame.draw.circle(self.screen, (0, 0, 255), (x, y), self.BALL_RADIUS)
        
    def set_target_position(self, target_position):
        self.target_position = target_position
        
    def reset(self):
        self.is_first_time = True
        self.prev_angle = 0
        self.ball_velocity = 0
        self.ball_position = None
        self.ball_distance_from_start = 5
        
        self.time_passed = 0
        
        self.eps_reward = 0
        
        return [self.ball_distance_from_start, self.target_position,self.ball_velocity,math.degrees(self.ANGLE1)], self.reward(), True

    def step(self, action):
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_DOWN]:
                self.ANGLE2 += math.radians(1)
            if keys[pygame.K_UP]:
                self.ANGLE2 -= math.radians(1)
        
        if self.ball_distance_from_start < 0 or self.ball_distance_from_start > self.RADIUS1 :
            return self.reset()
        
        self.ANGLE2 = self.prev_angle
        
        if action == 0:
            self.ANGLE2 += math.radians(1)
        elif action == 1:
            self.ANGLE2 -= math.radians(1)
        # Action 2 (or None) will do nothing
            
        self.ANGLE1 = self.ANGLE2 * self.RADIUS2 / self.RADIUS1
        
        if self.render:
            self.screen.fill((255, 255, 255))
        
        vertical_stick_1_top = self.draw_vertical_stick(self.VERTICAL_STICK_1_POSITION, self.HEIGHT1)
        
        rotating_stick_1_top = self.draw_rotating_stick(vertical_stick_1_top, self.RADIUS1, self.ANGLE1)
        
        vertical_stick_2_top = self.draw_vertical_stick(self.VERTICAL_STICK_2_POSITION, self.HEIGHT2)
        rotating_stick_2_top = self.draw_rotating_stick(vertical_stick_2_top, self.RADIUS2, self.ANGLE2)
        
        distance_btw_stick_tops = self.calculate_distance(rotating_stick_1_top, rotating_stick_2_top)
        
        if math.isclose(distance_btw_stick_tops, self.actual_distance, abs_tol=0.1):
            self.prev_angle = self.ANGLE2
            
        self.draw_connector(rotating_stick_1_top, rotating_stick_2_top)
        
        self.time_passed += self.dt
        
        if self.is_first_time:
            self.ball_position = list(vertical_stick_1_top)
            self.is_first_time = False
        else:
            #  neww ========================================
            vertical_inclination_angle = math.radians(90) - self.ANGLE1
            vertical_inclination_angle = int(math.degrees(vertical_inclination_angle))
            # print(vertical_inclination_angle)
            acceleration = self.g * math.cos(math.radians(vertical_inclination_angle))/(1+self.I)
            self.ball_velocity += acceleration * self.dt
            displacement = self.ball_velocity * self.dt
            self.ball_distance_from_start += displacement*100
            
            if vertical_inclination_angle == 90:  # Flat surface
                mu_k = 0.05  # Coefficient of kinetic friction
                friction_acc = mu_k * self.g  # constant deceleration
                if self.ball_velocity > 0:
                    self.ball_velocity -= friction_acc * self.dt
                    if self.ball_velocity < 0:
                        self.ball_velocity = 0
                elif self.ball_velocity < 0:
                    self.ball_velocity += friction_acc * self.dt
                    if self.ball_velocity > 0:
                        self.ball_velocity = 0
                acceleration = -friction_acc if self.ball_velocity != 0 else 0
            # # =========================================
            
            if self.render:
                self.ball_position = self.calculate_angular_displacement(vertical_stick_1_top, self.ball_distance_from_start, self.ANGLE1)
                
                stats = [
                    f"Time: {self.time_passed:.2f}s",
                    f"Velocity: {self.ball_velocity:.2f} m/s",
                    f"Acceleration: {acceleration:.2f} m/s²",
                    f"Incline Angle: {math.degrees(self.ANGLE1):.0f}°",
                    f"Ball distance from start: {self.ball_distance_from_start:.2f} m",
                    f"Target position: {self.target_position:.2f} m",
                ]
                
                for i, stat in enumerate(stats):
                    text = self.font.render(stat, True, (0, 0, 0))
                    self.screen.blit(text, (10, 10 + i * 20))
                
                # Draw target position indicator
                target_position = self.calculate_angular_displacement(vertical_stick_1_top, self.target_position, self.ANGLE1)
                pygame.draw.circle(self.screen, (255, 0, 0), 
                                  (target_position[0] + self.BALL_RADIUS, 
                                   target_position[1] - self.BALL_RADIUS), 
                                  5, 2)  # Empty circle with line width 2
        
        if self.render:
            self.draw_ball(self.ball_position)
            pygame.display.flip()
            self.clock.tick(60)
            
        self.eps_reward += self.reward()
        
        return [self.ball_distance_from_start, self.target_position,self.ball_velocity,math.degrees(self.ANGLE1)], self.reward(), False
        # return [self.ball_distance_from_start, self.target_position], self.reward(), False
        # return [self.ball_distance_from_start, self.target_position, math.degrees(self.ANGLE1)], self.reward(), False
    
    def reward(self):
        
        distance_error = self.target_position - self.ball_distance_from_start
        
        close_reward = 0
        away_penalty = 0
        rest_penality = 0
        
        if math.isclose(distance_error, 0, abs_tol=0.5) and math.isclose(self.ball_velocity, 0, abs_tol=0.1):
            close_reward = 1000
            away_penalty = 0
            
        elif distance_error < 0 and self.ball_velocity < 0:
            normalized_away_distance = abs(distance_error) / (self.RADIUS1 - self.target_position)
            close_reward = 3 * math.exp(3*normalized_away_distance-2) + 0.5

            away_penalty = 0

        elif distance_error < 0 and self.ball_velocity > 0:
            normalized_away_distance = abs(distance_error) / (self.RADIUS1 - self.target_position)
            away_penalty = 3 * math.exp(3*normalized_away_distance-2) + 0.5
            
            close_reward = 0
        
        elif distance_error > 0 and self.ball_velocity > 0:
            
            normalized_close_distance = distance_error / self.target_position
            close_reward = 3 * math.exp(-(3*normalized_close_distance-2)) + 0.5
            away_penalty = 0
            
        elif distance_error > 0  and self.ball_velocity < 0:
            normalized_away_distance = distance_error / self.target_position
            away_penalty = 3 * math.exp(3*normalized_away_distance-2) + 0.5
            close_reward = 0
            
        else:
            rest_penality = abs(distance_error)
            
        # print("reward : ", (close_reward - away_penalty - rest_penality))
        # print("episode reward : ", self.eps_reward)

        return float(close_reward - away_penalty - rest_penality)
