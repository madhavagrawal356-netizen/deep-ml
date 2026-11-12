def tick_bars(ticks: list, bar_size: int) -> list:
    """
    Sample tick data into tick bars.
    
    Args:
        ticks: List of tuples (timestamp, price, volume) representing individual trades
        bar_size: Number of ticks per bar
    
    Returns:
        List of dictionaries with keys: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    """
    if not ticks or bar_size<=0:
        return []
    start=0 
    end = start+bar_size-1 
    result = []
    while end<len(ticks):
        my_list = ticks[start:end+1] 
        _ , ope, _ = my_list[0]
        timestamp, clo, _ = my_list[-1]
        high, low = ope,clo
        volume=0
        for item in my_list:
            _, price,vol = item 
            volume+=vol 
            if price<low:
                low=price 
            if price>high:
                high=price 
        result.append({'timestamp': timestamp, 'open': ope, 'high': high, 'low':low 
        , 'close':clo, 'volume':volume})
        start = end+1 
        end = start+bar_size-1
    return result
        