import wisardpkg as wp

def circular_thermometer(num, size):
    if(not(size%2 == 0)):
        size = size+1        
    code = [0] * size

    if(num + size/2 < len(code)):
        for i in range(num, int(num + size/2)):
            code[i] = 1
    else:
        count = 0
        for i in range(num, len(code)):
            code[i] = 1
            count += 1
        for i in range(0, int(size/2 - count)):
            code[i] = 1
    
    return code

def sum_bin_input(b1, b2):
    b = []
    for i in range(len(b1)):
        b.append(b1[i])
    for i in range(len(b2)):
        b.append(b2[i])
        
    return wp.BinInput(b)

def circular_thermometer_total(bi, size):
    entry = wp.BinInput(circular_thermometer(bi[0], size))
    for i in range(1, len(bi)):
        entry = sum_bin_input(entry, wp.BinInput(circular_thermometer(bi[i], size)))
    return entry

print(circular_thermometer(0, 10))
print(circular_thermometer(4, 10))
print(circular_thermometer(9, 10))
print(circular_thermometer(241, 310))

print(circular_thermometer_total([1, 11, 41], 27))