area_code, prefix_number, postfix_number = [int(num) for num in input().split()]

print('Country  Phone Number')
print('-------  ------------')
print(f'U.S.     +1 ({area_code}){prefix_number}-{postfix_number}')
print(f'Brazil   +55 ({area_code}){prefix_number+100}-{postfix_number}')
print(f'Croatia  +385 ({area_code}){prefix_number}-{postfix_number+50}')
print(f'Egypt    +20 ({area_code+30}){prefix_number}-{postfix_number}')
print(f'France   +33 ({prefix_number}){area_code}-{postfix_number}')