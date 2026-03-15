import datetime

d1 = datetime.date(2002, 5, 9)
d2 = datetime.date(2003, 6, 22)

print(f"May 9, 2002 was a {d1.strftime('%A')}")
print(f"May 11, 2002 was a {(d1+datetime.timedelta(days=2)).strftime('%A')}")
print(f"June 22, 2003 was a {d2.strftime('%A')}")

sundays_june_2003 = [d for d in range(1, 31) if datetime.date(2003, 6, d).weekday() == 6]
print(f"Sundays in June 2003: {sundays_june_2003}")
print(f"4th Sunday of June 2003: {sundays_june_2003[3] if len(sundays_june_2003) > 3 else 'N/A'}")
