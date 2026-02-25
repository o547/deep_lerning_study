class Perceptron:
    def threshold(self, total):
        if total <= 0:
            return 0
        else:
            return 1

    def AND(self, x1, x2):
        w1 = 1
        w2 = 1
        b = -1
        total = x1 * w1 + x2 * w2 + b
        return self.threshold(total)

    def OR(self, x1, x2):
        w1 = 1
        w2 = 1
        b = 0
        total = x1 * w1 + x2 * w2 + b
        return self.threshold(total)

    def NAND(self, x1, x2):
        w1 = -2
        w2 = -2
        b = 3
        total = x1 * w1 + x2 * w2 + b
        return self.threshold(total)

    def XOR(self, x1, x2):
        a = self.OR(x1, x2)
        b = self.NAND(x1, x2)
        return self.AND(a, b)


# p = Perceptron()
# print("0, 0 = " + str(p.XOR(0, 0)))
# print("0, 1 = " + str(p.XOR(0, 1)))
# print("1, 0 = " + str(p.XOR(1, 0)))
# print("1, 1 = " + str(p.XOR(1, 1)))
