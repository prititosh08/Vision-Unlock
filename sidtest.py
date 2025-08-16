import winsound
import time

# YES sound
winsound.Beep(1000, 150)  # Medium-high, short
time.sleep(0.05)
winsound.Beep(1400, 200)  # Higher, slightly longer

winsound.Beep(600, 300)   # Low and long
time.sleep(0.05)
winsound.Beep(400, 400)   # Deeper, longer