class P1Random:
	def __init__(self):
		self._next = 0
		self._MULTIPLIER = 1103515245
		self._ADDEND = 12345
		self._DIVISOR = 65536

	def next_short(self, limit = 2**15-1):
		self._next = (self._next * self._MULTIPLIER + self._ADDEND) % 2**64
		if self._next > 2**63-1:
			self._next -= 2**64
		
		# using (next // DIVISOR) results in incorrect value for negative numbers
		value = int(self._next / self._DIVISOR) % 2**64
		if value > 2**63-1:
			value -= 2**64

		if value < 0:
			value = value % limit - limit
		else:
			value = value % limit
		
		return abs(value)

	def next_int(self, limit = 2**31-1):
		val = self.next_short() << 16
		val %= 2**32
		if val > 2**31-1:
			val -= 2**32
		return (val | self.next_short()) % limit
