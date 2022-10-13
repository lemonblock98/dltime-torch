
def solution(n, s):
    attack_list = [0] * (n+1)
    defence_list = [0] * (n+1)
    
    for i in range(1, n+1):
        if s[i-1] == '0':
            attack_list[i] = attack_list[i-1] + i
        else:
            attack_list[i] = attack_list[i-1]
    
    for i in range(n, 0, -1):
        if s[i-1] == '1':
            defence_list[i-1] = defence_list[i] + i
        else:
            defence_list[i-1] = defence_list[i]
    
    min_score = 1e10
    for i in range(n+1):
        if abs(attack_list[i] - defence_list[i]) < min_score:
            min_score = abs(attack_list[i] - defence_list[i])
    
    return min_score

n = 4
s = '0011'
print(solution(n, s))
