import regex
import re

def split(s):
    i = iter(re.split(r'(\(?\d+(?:\.\d+)+\)?|\(?\d+\)|\(?\b(?=M|(?:CM|CD|D?C)|(?:XC|XL|L?X)|(?:IX|IV|V?I))M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})[.)])', s, flags=re.IGNORECASE))
    return next(i) + '\n'.join(map(''.join, zip(i, i)))

str = "1. An orphaned child discovers they have magical powers and must learn to use them to save their kingdom. 2. A young adult embarks on a journey to find their long-lost family members. 3. A family moves to a new town and discovers secrets that have been hidden for centuries. 4. A mysterious virus turns people into zombies, and a group of survivors must find a way to survive. 5. A group of friends must work together to defeat an evil wizard who is trying to take over the world. 6. A young girl discovers she is a princess from another world and must learn to use her newfound powers to save her people. 7. A scientist invents a time machine, and the characters must use it to travel to the past in order to save the future. 8. A group of teenagers find themselves trapped in a mysterious realm filled with mystical creatures and must find a way to escape. 9. A family discovers that their new home is haunted by a sinister force. 10. A group of strangers find themselves in a strange world and must work together to find a way out."
out = split(str)

print(out)
