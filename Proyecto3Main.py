import copy
from dis import dis


class Node:

    def __init__(self, matriz, countGen):
        self.matriz = matriz
        self.countGen = countGen

    def addValues(self, matriz):
        self.matriz = matriz

    def getValues(self):
        return self.matriz

    def isEqual(self, nodo):
        if (self.matriz == nodo.getMatrix()):  # para sacar la matriz que esta en el nodo
            return 1
        else:
            return 0

    def getCountGen(self):
        return self.countGen

    def getMatrix(self):
        return self.matriz

    def __repr__(self):
        return str(self.matriz)
    def detectEmptySpace(self):
        return self.matriz.index(0)
    def getInvCount(self):
        inv_count = 0
        empty_value = -1
        for i in range(0, 9):
            for j in range(i + 1, 9):
                if self.matriz[j] != empty_value and self.matriz[i] != empty_value and self.matriz[i] > self.matriz[j]:
                    inv_count += 1
        return inv_count

    def isSolvable(self):
        inv_count = self.getInvCount([j for sub in self.matriz for j in sub])
        return (inv_count % 2 == 0)


class NodeAnalyzer:
    counterGen = 1

    def __init__(self, initNode, endNode):
        self.initNode = initNode
        self.current = initNode
        self.endNode = endNode

    def isFinal(self, current):
        return current.isEqual(self.endNode)

    def detectEmptySpace(self, node):
        try:
            return node.getValues().index(0)
        except:
            return -1

    def getAvailableMoves(self, node):
        emptyIndex = self.detectEmptySpace(node)
        if (emptyIndex == 0 or emptyIndex == 2 or emptyIndex == 6 or emptyIndex == 8):
            return 2
        if (emptyIndex == 1 or emptyIndex == 3 or emptyIndex == 5 or emptyIndex == 7):
            return 3
        if (emptyIndex == 4):
            return 4
        else:
            return -1

    def getAvailableMoves_on_matrix_with_zeros(self, matriz):
        return [i for i, x in enumerate(matriz) if x == 0]

    def change_matrix(self, empty_space, position_to_shift, matrix, node):
        matrix[empty_space] = matrix[position_to_shift]
        matrix[position_to_shift] = 0
        nodecount=node.getCountGen()+1
        newnode= Node(matrix, nodecount)
        return newnode

    def generateMoves_on_matrix_with_zeros(self, node):
        matriz = node.getMatrix()
        positions_empty_spaces = self.getAvailableMoves_on_matrix_with_zeros(node.getMatrix())
        moves_available = []
        for empty_space in positions_empty_spaces:
            if empty_space == 0:
                moves_available.append(self.change_matrix(empty_space, 1, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 3, copy.copy(matriz), node))
            elif empty_space == 1:
                moves_available.append(self.change_matrix(empty_space, 0, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 2, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 4, copy.copy(matriz), node))
            elif empty_space == 2:
                moves_available.append(self.change_matrix(empty_space, 1, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 5, copy.copy(matriz), node))
            elif empty_space == 3:
                moves_available.append(self.change_matrix(empty_space, 0, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 4, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 6, copy.copy(matriz), node))
            elif empty_space == 4:
                moves_available.append(self.change_matrix(empty_space, 1, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 5, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 7, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 3, copy.copy(matriz), node))
            elif empty_space == 5:
                moves_available.append(self.change_matrix(empty_space, 2, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 4, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 8, copy.copy(matriz), node))
            elif empty_space == 6:
                moves_available.append(self.change_matrix(empty_space, 3, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 7, copy.copy(matriz), node))
            elif empty_space == 7:
                moves_available.append(self.change_matrix(empty_space, 6, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 4, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 8, copy.copy(matriz), node))
            else:
                moves_available.append(self.change_matrix(empty_space, 5, copy.copy(matriz), node))
                moves_available.append(self.change_matrix(empty_space, 7, copy.copy(matriz), node))
        return moves_available

    def generateMoves(self, node):
        emptyspaceIndex = self.detectEmptySpace(node)
        movesavailable = self.getAvailableMoves(node)
        nodeArray = []
        if (emptyspaceIndex == 0 and movesavailable == 2):
            # Falta chequear en priorityqueue si los movimientos ya estan hechos
            # movimiento hacia la izquierda esquina derecha
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            temp = matrizActual1[0]
            matrizActual1[0] = matrizActual1[1]
            matrizActual1[1] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp2 = matrizActual2[0]
            matrizActual2[0] = matrizActual2[3]
            matrizActual2[3] = temp2
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
        if (emptyspaceIndex == 1 and movesavailable == 3):
            # Falta chequear en priorityqueue si los movimientos ya estan hechos
            # movimiento hacia la derecha segunda posicion
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            matrizActual3 = copy.copy(matrizActual1)
            temp = matrizActual1[1]
            matrizActual1[1] = matrizActual1[0]
            matrizActual1[0] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp = matrizActual2[1]
            matrizActual2[1] = matrizActual2[4]
            matrizActual2[4] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
            temp = matrizActual3[1]
            matrizActual3[1] = matrizActual3[2]
            matrizActual3[2] = temp
            newNode3 = Node(matrizActual3, node.getCountGen() + 1)
            nodeArray.append(newNode3)
        if (emptyspaceIndex == 2 and movesavailable == 2):
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            temp = matrizActual1[2]
            matrizActual1[2] = matrizActual1[1]
            matrizActual1[1] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            # movimiento hacia arriba esquina derecha
            temp = matrizActual2[2]
            matrizActual2[2] = matrizActual2[5]
            matrizActual2[5] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
        if (emptyspaceIndex == 3 and movesavailable == 3):
            # Falta chequear en priorityqueue si los movimientos ya estan hechos
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            matrizActual3 = copy.copy(matrizActual1)
            temp = matrizActual1[3]
            matrizActual1[3] = matrizActual1[0]
            matrizActual1[0] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp = matrizActual2[3]
            matrizActual2[3] = matrizActual2[4]
            matrizActual2[4] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
            temp = matrizActual3[3]
            matrizActual3[3] = matrizActual3[6]
            matrizActual3[6] = temp
            newNode3 = Node(matrizActual3, node.getCountGen() + 1)
            nodeArray.append(newNode3)
        if (emptyspaceIndex == 5 and movesavailable == 3):
            # Falta chequear en priorityqueue si los movimientos ya estan hechos
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            matrizActual3 = copy.copy(matrizActual1)
            temp = matrizActual1[5]
            matrizActual1[5] = matrizActual1[2]
            matrizActual1[2] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp = matrizActual2[5]
            matrizActual2[5] = matrizActual2[4]
            matrizActual2[4] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            newNode2.addValues(matrizActual2)
            nodeArray.append(newNode2)
            temp = matrizActual3[5]
            matrizActual3[5] = matrizActual3[8]
            matrizActual3[8] = temp
            newNode3 = Node(matrizActual3, node.getCountGen() + 1)
            nodeArray.append(newNode3)
        if (emptyspaceIndex == 6 and movesavailable == 2):
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            temp = matrizActual1[6]
            matrizActual1[6] = matrizActual1[3]
            matrizActual1[3] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp = matrizActual2[6]
            matrizActual2[6] = matrizActual2[7]
            matrizActual2[7] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
        if (emptyspaceIndex == 7 and movesavailable == 3):
            # Falta chequear en priorityqueue si los movimientos ya estan hechos
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            matrizActual3 = copy.copy(matrizActual1)
            temp = matrizActual1[7]
            matrizActual1[7] = matrizActual1[6]
            matrizActual1[6] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp = matrizActual2[7]
            matrizActual2[7] = matrizActual2[4]
            matrizActual2[4] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
            temp = matrizActual3[7]
            matrizActual3[7] = matrizActual3[8]
            matrizActual3[8] = temp
            newNode3 = Node(matrizActual3, node.getCountGen() + 1)
            nodeArray.append(newNode3)
        if (emptyspaceIndex == 8 and movesavailable == 2):
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            temp = matrizActual1[8]
            matrizActual1[8] = matrizActual1[5]
            matrizActual1[5] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp = matrizActual2[8]
            matrizActual2[8] = matrizActual2[7]
            matrizActual2[7] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
        if (emptyspaceIndex == 4 and movesavailable == 4):
            matrizActual1 = node.getValues()
            matrizActual2 = copy.copy(matrizActual1)
            matrizActual3 = copy.copy(matrizActual1)
            matrizActual4 = copy.copy(matrizActual1)
            temp = matrizActual1[4]
            matrizActual1[4] = matrizActual1[1]
            matrizActual1[1] = temp
            newNode1 = Node(matrizActual1, node.getCountGen() + 1)
            nodeArray.append(newNode1)
            temp = matrizActual2[4]
            matrizActual2[4] = matrizActual2[3]
            matrizActual2[3] = temp
            newNode2 = Node(matrizActual2, node.getCountGen() + 1)
            nodeArray.append(newNode2)
            temp = matrizActual3[4]
            matrizActual3[4] = matrizActual3[5]
            matrizActual3[5] = temp
            newNode3 = Node(matrizActual3, node.getCountGen() + 1)
            nodeArray.append(newNode3)
            temp = matrizActual4[4]
            matrizActual4[4] = matrizActual4[7]
            matrizActual4[7] = temp
            newNode4 = Node(matrizActual4, node.getCountGen() + 1)
            nodeArray.append(newNode4)
        return nodeArray


def getManhattanCost(matriz):
    manhattanDistance = 0
    for i, item in enumerate(matriz):
        prev_row, prev_col = int(i / 3), i % 3
        goal_row, goal_col = int(item / 3), item % 3
        manhattanDistance += abs(prev_row - goal_row) + abs(prev_col - goal_col)
    return manhattanDistance


class PriorityQueue(object):
    queue = []

    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def isEmpty(self):
        return len(self.queue) == 0
    def incert(self, node):
        if len(self.queue) == 0:
            self.queue.append(node)
        else:
            for x in range(0, len(self.queue)):
                costOG = (getManhattanCost(self.queue[x].getMatrix()) + node.getCountGen())
                costN = (getManhattanCost(node.getMatrix()) + node.getCountGen())
                if costOG > costN:
                    self.queue.insert(x, node)
                    return 1
                else:
                    self.queue.append(node)
                    return 0

    def pop(self):
        return self.queue.pop(0)

    def __contains__(self, node):
        if len(self.queue) == 0:
            return 0
        else:
            for x in range(0, len(self.queue)):

                if self.queue[x].getMatrix() == node.getMatrix():
                    return True
                else:
                    return False

    def flush(self):
        self.queue = []


def a(initNode, endNode):
    na = NodeAnalyzer(initNode, endNode)
    pq = PriorityQueue()
    pq.incert(initNode)
    explored = []
    counter = 0

    while True:
        if pq.isEmpty():
            return "Fail"
        current = pq.pop()
        print("current: ", current.getMatrix())
        # pq.flush() por si necesitamos vaciar el pq cada vez que saca el menor
        if na.isFinal(current):
            return "Success ", current.getMatrix(), "  Nodes generated:", current.getCountGen()
        explored.append([v for v in current.getMatrix()])
        children = na.generateMoves(current)
        for child in children:
            copyChild = copy.copy(child.getMatrix())
            copyNode = copy.copy(child)
            if copyChild not in explored or pq.__contains__(copyNode):
                pq.incert(child)
                counter += 1


def aForPattern(initNode, endNode):
    na = NodeAnalyzer(initNode, endNode)
    pq = PriorityQueue()
    pq.incert(initNode)
    explored = []
    counter = 0

    while True:
        if pq.isEmpty():
            return "Fail"
        current = pq.pop()
        print("current: ", current.getMatrix())
        # pq.flush() por si necesitamos vaciar el pq cada vez que saca el menor
        if na.isFinal(current):
            return "Success: ", current.getMatrix(), "Nodes generated: ", current.getCountGen()
        explored.append([v for v in current.getMatrix()])
        children = na.generateMoves_on_matrix_with_zeros(current)
        for child in children:
            copyChild = copy.copy(child.getMatrix())
            copyNode = copy.copy(child)
            if copyChild not in explored or pq.__contains__(copyNode):
                pq.incert(child)
                counter += 1


class DisjointDB:
    def __init__(self, matriz, filename):
        self.matriz = copy.copy(matriz)
        self.filename = filename
        self.database = {}

    # Permutaciones de una lista obtenido de STACKOVERFLOW
    def createPermutation(self, matriz):
        if len(matriz) == 0: return []
        if len(matriz) == 1: return [matriz]
        l = []
        for i in range(len(matriz)):
            m = matriz[i]
            remainingLst = matriz[:i] + matriz[i + 1:]

            for p in self.createPermutation(remainingLst):
                l.append([m] + p)
        return l

    # isUnsolvable de StackOverflow
    def isUnsolvable(self, matriz):
        counter = 0
        for i in range(8):
            for j in range(i, 9):
                if (matriz[i] > matriz[j] and matriz[j] != 0):
                    counter += 1
        if counter % 2 == 1:
            return True
        else:
            return False

    def getDatabaseRow(self, matriz):
        firstPattern = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        secondPattern = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, element in enumerate(matriz):
            if int(element) <= 4:
                firstPattern[i] = int(element)
            else:
                secondPattern[i] = int(element)
        return firstPattern, secondPattern, getManhattanCost(firstPattern), getManhattanCost(secondPattern)

    def toStringlist(self, matriz):
        stringMatriz = ''
        for element in matriz:
            stringMatriz += str(element)
        return stringMatriz

    def databaseCreation(self):
        open(self.filename, 'w').close()
        with open(self.filename, 'a') as databaseFile:  ##Se hace a para poner en modo append
            for permutation in self.createPermutation(self.matriz):
                if not self.isUnsolvable(permutation):
                    firstPattern, secondPattern, costPattern1, costPattern2 = self.getDatabaseRow(permutation)
                    start = Node(firstPattern, 0)
                    end = Node([0, 1, 2, 3, 4, 0, 0, 0, 0], 0)
                    costo1 = aForPattern(start, end)
                    start2 = Node(firstPattern, 0)
                    end2 = Node([0, 0, 0, 0, 0, 5, 6, 7, 8], 0)
                    costo2 = aForPattern(start2, end2)
                    databaseFile.write(
                        f'{self.toStringlist(permutation)},{self.toStringlist(firstPattern)},{costo1},{self.toStringlist(secondPattern)},{costo2}\n')

    def gettxttoConsole(self):
        dbfile = open(self.filename, 'r')  # Se hace r para poner en modo read.
        for line in dbfile:
            lineList = line.split(',')
            totalCost = int(lineList[2]) + int(lineList[4])
            self.database[str([int(i) for i in lineList[0]])] = totalCost

    def searchPattern(self, pattern: list):
        return self.database.get(str(pattern))


def getMovesDone(previa, current):
    previa = copy.copy(previa.getValues())
    current = copy.copy(current.getValues())
    if (previa.detectEmptySpace() == 0 and current.detectEmptySpace() == 3) or (
            previa.detectEmptySpace() == 1 and current.detectEmptySpace() == 4) or (
            previa.detectEmptySpace() == 2 and current.detectEmptySpace() == 5) or (
            previa.detectEmptySpace() == 3 and current.detectEmptySpace() == 6) or (
            previa.detectEmptySpace() == 4 and current.detectEmptySpace() == 7) or (
            previa.detectEmptySpace() == 5 and current.detectEmptySpace() == 8):
        return "UP"
    elif (previa.detectEmptySpace() == 3 and current.detectEmptySpace() == 0) or (
            previa.detectEmptySpace() == 4 and current.detectEmptySpace() == 1) or (
            previa.detectEmptySpace() == 5 and current.detectEmptySpace() == 2) or (
            previa.detectEmptySpace() == 6 and current.detectEmptySpace() == 3) or (
            previa.detectEmptySpace() == 7 and current.detectEmptySpace() == 4) or (
            previa.detectEmptySpace() == 8 and current.detectEmptySpace() == 5):
        return "DOWN"
    elif (previa.detectEmptySpace() == 0 and current.detectEmptySpace() == 1) or (
            previa.detectEmptySpace() == 1 and current.detectEmptySpace() == 2) or (
            previa.detectEmptySpace() == 3 and current.detectEmptySpace() == 4) or (
            previa.detectEmptySpace() == 4 and current.detectEmptySpace() == 5) or (
            previa.detectEmptySpace() == 6 and current.detectEmptySpace() == 7) or (
            previa.detectEmptySpace() == 7 and current.detectEmptySpace() == 8):
        return "LEFT"
    elif (previa.detectEmptySpace() == 1 and current.detectEmptySpace() == 0) or (
            previa.detectEmptySpace() == 2 and current.detectEmptySpace() == 1) or (
            previa.detectEmptySpace() == 4 and current.detectEmptySpace() == 3) or (
            previa.detectEmptySpace() == 5 and current.detectEmptySpace() == 4) or (
            previa.detectEmptySpace() == 7 and current.detectEmptySpace() == 6) or (
            previa.detectEmptySpace() == 8 and current.detectEmptySpace() == 7):
        return "RIGHT"
    else:
        return ".."


start = Node([6, 5, 1, 4, 2, 3, 0, 8, 7], 0)
goal = Node([0, 1, 2, 3, 4, 5, 6, 7, 8], 0)

db = DisjointDB(start.getValues(), 'database.txt')
# db.databaseCreation()
# db.gettxttoConsole()

print("---- USING MANHATTAN HEURISTIC ----")
print(a(start, goal))
print("---- USING DB HEURISTIC ----")
start2 = Node([0, 0, 1, 4, 2, 3, 0, 0, 0], 0)
goal2 = Node([0, 1, 2, 3, 4, 0, 0, 0, 0], 0)
print(aForPattern(start2, goal2))
