import math



def fcfs_preemption(requests, num_blocks, block_size):

    n = len(requests)

    finish = [None] * n
    preemptions = [0] * n

    free_blocks = num_blocks
    waiting = []
    swapped = []

    peak_swapped_blocks = 0

    # Initialize request states
    state = []

    for arrival, prompt, output in requests:
        state.append({
            'arrival': arrival,
            'prompt': prompt,
            'output': output,
            'cache': 0,
            'output_count': 0,
            'blocks': 0,
            'status': 'not_arrived'
        })

    # Calculate blocks required for a given number of tokens
    def get_blocks(tokens):
        return (tokens + block_size - 1) // block_size

    t = 0

    while any(s['status'] != 'done' for s in state):

        # --------------------------------
        # 1. ARRIVALS
        # --------------------------------
        for i in range(n):
            if (state[i]['status'] == 'not_arrived'
                    and state[i]['arrival'] <= t):

                state[i]['status'] = 'waiting'
                waiting.append(i)

        # --------------------------------
        # 2. SWAP IN
        # --------------------------------
        while swapped:

            i = swapped[0]
            required = state[i]['blocks']

            if required <= free_blocks:
                swapped.pop(0)

                free_blocks -= required
                state[i]['status'] = 'running'

            else:
                break

        # --------------------------------
        # 3. ADMISSION
        # --------------------------------
        admitted_this_step = set()

        if not swapped:

            while waiting:

                i = waiting[0]
                required = get_blocks(state[i]['prompt'])

                if required <= free_blocks:

                    waiting.pop(0)

                    free_blocks -= required

                    state[i]['status'] = 'running'
                    state[i]['cache'] = state[i]['prompt']
                    state[i]['output_count'] = 1
                    state[i]['blocks'] = required

                    admitted_this_step.add(i)

                else:
                    break

        # --------------------------------
        # 4. DECODE
        # --------------------------------
        for i in range(n):

            if state[i]['status'] != 'running':
                continue

            # Newly admitted requests already ran their prompt step
            if i in admitted_this_step:
                continue

            # Check whether another block is needed
            if state[i]['cache'] % block_size == 0:

                # Preempt if GPU has no free blocks
                if free_blocks == 0:

                    # Find latest-arrived running request
                    victim = max(
                        j for j in range(n)
                        if state[j]['status'] == 'running'
                    )

                    # Swap out the entire request
                    state[victim]['status'] = 'swapped'

                    preemptions[victim] += 1

                    free_blocks += state[victim]['blocks']

                    swapped.append(victim)
                    swapped.sort()

                    # If the current request was preempted,
                    # it cannot decode during this step.
                    if victim == i:
                        continue

                # Allocate the required block
                free_blocks -= 1
                state[i]['blocks'] += 1

            # Generate one output token
            state[i]['cache'] += 1
            state[i]['output_count'] += 1

        # --------------------------------
        # PEAK SWAPPED BLOCKS
        # --------------------------------
        current_swapped_blocks = sum(
            state[i]['blocks'] for i in swapped
        )

        peak_swapped_blocks = max(
            peak_swapped_blocks,
            current_swapped_blocks
        )

        # --------------------------------
        # 5. FINISH
        # --------------------------------
        for i in range(n):

            if (state[i]['status'] == 'running'
                    and state[i]['output_count'] == state[i]['output']):

                free_blocks += state[i]['blocks']

                state[i]['blocks'] = 0
                state[i]['status'] = 'done'

                finish[i] = t

        t += 1

    return {
        'finish': finish,
        'preemptions': preemptions,
        'peak_swapped_blocks': peak_swapped_blocks
    }

    
