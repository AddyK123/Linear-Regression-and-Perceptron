alpha: 2.5, iteration number 200

I chose this hyperparameter pair as my ideal through a variety of methods and judged this to be the most fit. Firstly, I created a cost function that was used in testing the “fit” of each hyperparameter pair. This was initially included within the iterations of each alpha value and helped me gain a sense of what learning rates we working best for modeling the data. 

After looking at the printed list of each cost function and learning rate pair in the terminal I saw that we were finding a local minimum between .5 and 5. 1 was the most fit of all other learning rates and so naturally the value had to be around there and within the brackets defined.

So I set about testing whole numbers between 0 and 5 to get a better sense of the progression we were having, when I saw that two was the lowest I set the boundaries to [1, 3] and began testing learning rates at .5 intervals within the range. I kept this process of minimizing the possible range of learning rates and testing smaller and smaller intervals until I was at [2.45, 2.55] and .05 intervals. At this point I knew my learning rate of 2.5 was more or less ideal for the data it was operating on.

Now that I had found the ideal learning rate, I began testing multiple iteration numbers to find my ideal n value. Looking at the current cost function outputs, I thought that an increase in the n value would help so created an array of iterations [100, 150, 200, 250, 500, 1000] and tested them on alpha values of [2.4, 2.5, 2.6]. This meant I had 18 data points to see the cost function outputs and make sure that I was finding the ideal n value. I saw that the minimum within this initial set was 200 and so set the bounds to [150, 250] and tested in intervals of 25. Similar to the learning rate process, through slow refinement I was able to find that 200 was the ideal n value for modeling the data the gradient descent was operating on.



