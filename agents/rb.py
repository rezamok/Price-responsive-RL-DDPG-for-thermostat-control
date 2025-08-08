class RBAgent:
    """
    Very basic Rule-Based agent heating or cooling only as soon as the bounds are reached.
    """

    def __init__(self, data_kwargs, model_kwargs, agent_kwargs, umar_model=None, battery_model=None,
                 compute_reward=compute_reward, room = '272'):

        self.simple_env = agent_kwargs["simple_env"]

        print("\nPreparing the rule-based agent")
        #all_inputs, all_outputs, base_indices, effect_indices = build_physics_based_inputs_outputs_indices()

        # all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
        #               'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Power 272', 'Valve 272', 'Case']
        #room = 272
        if room == '274':
            base_indices = [1, 4, 7, 8, 9, 10, 11]
            effect_indices = [2, 4, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 274', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 274', 'Case']
            all_outputs = ['Temperature 274']
        elif room == '272':
            base_indices = [1, 3, 7, 8, 9, 10, 11]
            effect_indices = [2, 3, 5, 6]
            all_inputs = ['Solar irradiation', 'Outside temperature', 'Temperature 272', 'Temperature 273', 'Month sin',
                          'Month cos', 'Weekday', 'Timestep sin', 'Timestep cos', 'Valve 272', 'Case']
            all_outputs = ['Temperature 272']
        if (umar_model is None) | (battery_model is None):
            # Prepare the battery and UMAR models
            if self.simple_env:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)
            else:
                umar_model, model_kwargs = prepare_model_mina(agent_kwargs=agent_kwargs,
                                                              Y_columns=all_outputs,
                                                              X_columns=all_inputs,
                                                              base_indices=base_indices,
                                                              effect_indices=effect_indices,
                                                              room=room)

        # Prepare the environment
        if self.simple_env:
            env = ToyEnv(umar_model=umar_model,
                         battery_model= None,
                         agent_kwargs=agent_kwargs,
                         compute_reward=compute_reward)
        else:
            env = UMAREnv(umar_model=umar_model,
                          battery_model=battery_model,
                          agent_kwargs=agent_kwargs,
                          compute_reward=compute_reward)

        self.small_model = agent_kwargs["small_model"]
        self.compute_reward = compute_reward

        self.her = agent_kwargs["her"]

        # Save the environment
        self.env = env
        self.half_degree_scaled = (self.env.scaled_temp_bounds[1, :] - self.env.scaled_temp_bounds[0, :]) / 2 / (
                self.env.temp_bounds[1] - self.env.temp_bounds[0])
        # (np.mean(self.env.scaled_temp_bounds, axis=0) - self.env.scaled_temp_bounds[0, :]) / 2

    def take_decision(self, observation, goal=None):
        """
        Decision making function: heat when the temperature drops below the lower bound and cool when it
        goes above the upper bound.

        Args:
            observation: the current observation of the environment from which to take the decision
        """
        if goal is None:
            if len(self.env.temp_bounds) == 2:
                # print("mmmmm")
                # print(self.env.scaled_temp_bounds)
                # print(self.half_degree_scaled)
                # print((self.env.temp_bounds[1] - self.env.temp_bounds[0]))
                # print(len(self.env.rooms))
                # print(observation[-4] > 0.89999)
                if observation[-5] > 0.89999:
                    goal = self.env.scaled_temp_bounds[0, :] + self.half_degree_scaled
                else:
                    goal = self.env.scaled_temp_bounds[1, :] - self.half_degree_scaled

            if len(self.env.temp_bounds) == 4:
                if observation[-5] > 0.49999:
                    if observation[-4] < 0.10001:
                        goal = self.env.scaled_temp_bounds[0, :] + self.half_degree_scaled
                    elif observation[-4] < 0.49:
                        goal = self.env.scaled_temp_bounds[1, :] + self.half_degree_scaled
                    else:
                        raise NotImplementedError(f"What? Should be lower bound, lower than 0.5: {observation[-3]}")
                else:
                    if observation[-3] > 0.89999:
                        goal = self.env.scaled_temp_bounds[3, :] - self.half_degree_scaled
                    elif observation[-3] > 0.51:
                        goal = self.env.scaled_temp_bounds[2, :] - self.half_degree_scaled
                    else:
                        raise NotImplementedError(f"What? Should be higher bound, higher than 0.5: {observation[-3]}")

        if self.small_model:
            index = 8 if self.env.simple else 9
        else:
            index = 8 if self.env.simple else 10

        # Check in which case we are
        if observation[-5] > 0.4999:
            # print("xeee")
            # print(-(index + 2 * len(self.env.rooms)))
            # print(observation)
            # print( observation[-(index + 2 * len(self.env.rooms)):
            #                                -(index + len(self.env.rooms))])
            # print((goal > observation[-(index + 2 * len(self.env.rooms)):
            #                                -(index + len(self.env.rooms))]))
            # Heating case: Heat if we are below the lower bound
            action = np.array((goal >
                               observation[-(index + 2 * len(self.env.rooms)):
                                           -(index + len(self.env.rooms))]) * 1 * 0.8 + 0.1)
        elif observation[-5] < 0.10001:
            # Cooling: cool if we are above the upper bound
            action = np.array((goal <
                               observation[-(index + 2 * len(self.env.rooms)):
                                           -(index + len(self.env.rooms))]) * 1 * 0.8 + 0.1)

        else:
            raise NotImplementedError(f"The case shouldn't be {observation[-4]}")

        # Decide what to do of the battery
        if self.env.battery:
            electricity_consumption = self.env.umar_model.min_["Electricity total consumption"] + (
                    observation[self.env.elec_column] - 0.1) * (
                                              self.env.umar_model.max_["Electricity total consumption"] -
                                              self.env.umar_model.min_[
                                                  "Electricity total consumption"]) / 0.8
            energies = observation[-(3 + len(self.env.rooms)):-3]
            energy = sum([self.env.umar_model.min_[f"Energy room {room}"] + (energies[i] - 0.1) * (
                    self.env.umar_model.max_[f"Energy room {room}"] - self.env.umar_model.min_[
                f"Energy room {room}"]) / 0.8 for i, room in enumerate(self.env.rooms)])

            consumption = energy / self.env.COP + electricity_consumption

            if consumption > 0:
                # Transform in kWh, then in kW over 15 min
                margin = 0.96 / self.env.battery_size * (
                        observation[-1] - self.env.battery_margins[0] - 0.25) * 60 / self.env.umar_model.interval
                if margin > consumption:
                    action = np.append(action, max(-consumption, -self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(-margin, -self.env.battery_max_power), self.env.battery_max_power))
            else:
                margin = 0.96 / self.env.battery_size * (
                        self.env.battery_margins[1] - observation[-1] - 0.25) * 60 / self.env.umar_model.interval
                if margin > -consumption:
                    action = np.append(action, min(-consumption, self.env.battery_max_power))
                else:
                    action = np.append(action,
                                       min(max(margin, -self.env.battery_max_power), self.env.battery_max_power))

        # Return the action
        return action

    def run(self, sequence, goal_number=None, init_temp=None, render=True):
        """
        Run the agent over a sequence of inputs

        Args:
            sequence: a sequence of indices to get the data. If None, a random sequence is used
            render:   whether to plot the result or not
        """

        # Reset the environment to get the first observation
        observation = self.env.reset(sequence, goal_number, init_temp)
        temperatures = [self.env.scale_back_temperatures(observation[2])]
        actions = []
        if self.her:
            observation = observation['observation']
            goal = self.env.desired_goals[self.env.goal_number]
        else:
            goal = None

        done = False
        cumul_reward = 0
        length = 0
        k = 0
        # Iterate until the episode is over (i.e. the sequence is finished)
        while not done:
            # Choose an action
            action = self.take_decision(observation, goal)
            k = k+1
            # print("while")
            # print(action.dtype)
            action = torch.Tensor(action)
            # print(action.dtype)
            # print(observation.dtype)
            # Take a step
            observation, reward, done,_ = self.env.step(action, rule_based=True)
            # observation, reward, done, info = self.env.step(action, rule_based=True)
            temperatures.append(self.env.scale_back_temperatures(observation[2]))
            actions.append(action)
            if self.her:
                observation = observation['observation']
            # Recall the rewards
            cumul_reward += reward
            length += 1

        # Some modifications needed to be compatible with the 'render' function
        self.env.current_step -= 1
        if self.env.battery:
            self.env.battery_powers.pop()
            self.env.battery_soc.pop()
        self.env.electricity_imports.pop()

        # If wanted, plot the result
        if render:
            self.env.render()

        return cumul_reward, length, temperatures, actions
