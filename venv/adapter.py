


class adapter():
    def __init__(self,problem):
        self.state_to_number=dict()
        self.number_to_state=dict()
        self.state_counter=0

        self.action_to_number = dict()
        self.number_to_action = dict()
        self.action_counter = 0

        self.obs_to_number = dict()
        self.number_to_obs = dict()
        self.obs_counter = 0
        self.actionslist=[]
        self.actionslistindexes=[]
        self.obslist=[]
        self.obslistindexes=[]
        self.problem=problem
        self.sep_rewards=False

    def initstates(self,listofstates):
        """
        :param listofstates:
        purpose: build the dicts,
        state_to_number = (state) -> his number
        number_to_state = number -> [state]   note one is tuple of tuples and one is list of tuples.
        return: the number of init states
        """
        for state in listofstates:
            self.state_to_number[tuple(state)]=self.state_counter
            self.number_to_state[self.state_counter]=state
            self.state_counter += 1
        return len(self.number_to_state.keys())

    def stateToNumber(self,state):
        """
        :param state:
        :return: number that represent that state if its new state add to dict
        """
        number_of_state=self.state_to_number.get(tuple(state))
        if number_of_state is None:
            self.state_to_number[tuple(state)]=self.state_counter
            self.number_to_state[self.state_counter] = state
            number_of_state=self.state_counter
            self.state_counter += 1
        return number_of_state
    def numberToState(self,number):
        """
        :param number:
        :return: the state that the number represnt raise error if not found
        """
        state=self.number_to_state.get(number)
        if state is None:
            raise Exception('error in adapter, cannot find the state for this number')
        return state

    def initacitonsobs(self,actionslist,obslist):
        self.actionslist=actionslist
        self.actionslistindexes=list(range(0,len(actionslist)))
        self.obslist=obslist
        self.obslistindexes=list(range(0,len(obslist)))
        for action in actionslist:
            self.action_to_number[action]=self.action_counter
            self.number_to_action[self.action_counter]=action
            self.action_counter += 1
        for obs in obslist:
            self.obs_to_number[obs]=self.obs_counter
            self.number_to_obs[self.obs_counter]=obs
            self.obs_counter += 1

        return len(self.number_to_action.keys()),len(self.number_to_obs.keys())

    def actiontoNumber(self,action):
        return self.action_to_number[action]
    def numbertoAction(self,numberofaction):
        return self.number_to_action[numberofaction]
    def obstoNumber(self,obs):
        return self.obs_to_number[obs]
    def numbertoObservation(self,numberofobs):
        return self.number_to_obs[numberofobs]

    def blackbox(self, state, action):
        """
        get the call from the POMCP and adapt it to the black box so getting number and delegate it to balck box
        with the suitable state,action
        after getting the result from the black box adapt it to the pomcp meaning summ the rewards and turn state and obs to numbers.
        :param state:
        :param action:
        :return:
        """
        blackbox_state=self.numberToState(state)
        blackbox_action=self.numbertoAction(action)
        b_next_state,b_obs,b_rewards=self.problem.blackbox(blackbox_state,blackbox_action)
        pomcp_nextstate=self.stateToNumber(b_next_state)
        pomcp_obs=self.obstoNumber(b_obs)
        pomcp_reward=0
        if self.sep_rewards==True:
            return (pomcp_nextstate,pomcp_obs,b_rewards)
        for r in b_rewards:
            pomcp_reward+=r
        return (pomcp_nextstate,pomcp_obs,pomcp_reward)

    """def validactionsforrollout(self,state):
        return self.problem.validactionsforrollout(self.numberToState(state))"""










