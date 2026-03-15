import datetime

d = datetime.date(2003, 6, 22)
print(f"June 22, 2003 was a {d.strftime('%A')}")

sundays = [datetime.date(2003, 6, day) for day in range(1, 30) 
           if datetime.date(2003, 6, day).weekday() == 6]
print("Sundays in June 2003:", [s.day for s in sundays])

if d in sundays:
    print(f"June 22 is the {sundays.index(d)+1}th Sunday of June 2003")
else:
    print("Not a Sunday")

# Check May 2002
may9 = datetime.date(2002, 5, 9)
may10 = datetime.date(2002, 5, 10)
may11 = datetime.date(2002, 5, 11)
print(f"\nMay 9, 2002: {may9.strftime('%A')}")
print(f"May 10, 2002: {may10.strftime('%A')}")
print(f"May 11, 2002: {may11.strftime('%A')}")
