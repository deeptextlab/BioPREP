from datetime import timedelta

# function that shows the iteration process
def good_update_interval(total_iters, num_desired_updates):
    exact_interval = total_iters / num_desired_updates

    order_of_mag = len(str(total_iters)) - 1
    round_mag = order_of_mag - 1

    update_interval = int(round(exact_interval, -round_mag))

    if update_interval == 0:
        update_interval = 1

    return update_interval


# transform sec to hr
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(timedelta(seconds=elapsed_rounded))