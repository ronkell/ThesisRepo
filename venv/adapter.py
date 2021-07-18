


class adapter():
    def __init__(self):
        self.state_to_number=dict()
        self.number_to_state=dict()
        self.state_counter=0

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


