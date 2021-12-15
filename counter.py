
# DEFINITION: ROI-Counter=the space between two defined lines that will be the space where a id must be for a 
#             determined time to be incremented in the counter
class Counter:
    def __init__(self, max_limmit, min_counter, max_counter) -> None:
        """
        :param max_limmit: It says about what is the maximum amount of frames that an id can be countable,
                           if the distance between the first frame and the current frame is larger than max_limmit
                           that id will not be trackable.
        :param min_counter: the minimum amount of times that an id should be in the ROI-counter to be counted
        :param max_counter: the maximum amount of times that an id should be in the ROI-counter to be counter
        """
        self.max_limmit = max_limmit
        self.min_counter = min_counter
        self.max_counter = max_counter
        self.count = 0
        self.ids = {}
        self.counted_ids = {}

    def update(self, points, frame_counter):
        """
        This function update the points in the ids dictionary
        it receives only the points between the interest lines
        """

        for p in points:
            if p not in self.ids and p not in self.counted_ids:
                self.ids[p] = [1,frame_counter]
            elif p in self.ids:
                self.ids[p][0] +=1

                if self.ids[p][0] >= self.min_counter:
                    self.count+=1
                    self.counted_ids[p] = self.ids[p]
                    del self.ids[p]
                
                # if is too much time into the ROI-counter
                # if (frame_counter-self.ids[p][1]) >= self.max_limmit:
                    # del self.counted_ids[p]

            if p in self.counted_ids:
                self.counted_ids[p][0] +=1
                if self.counted_ids[p][0] >=self.max_counter:
                    self.count-=1
                    del self.counted_ids[p]

                # elif (frame_counter-self.counted_ids[p][1]) >= self.max_limmit:
                #     del self.counted_ids[p]    
