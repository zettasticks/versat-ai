from versatDefs import *
from versatCommons import *

import itertools

# Memory allocation is precomputed by transforming it into rectangle fitting,
# where the amount of memory is the width and the allocation time is the height and it is fixed

# TODO: How do we handle graphs that contain variable dimensions?
#       First we can separate constant size expressions and variable sized into two regions.
#       Second we should be able to generate an offset that depends on the dimension described.
# NOTE: If we find out that the size of operations is substantially different functions,
#       like one operation uses N memory while the other uses N^2 memory, then we can
#       probably solve this problem by further dividing the memory regions into one section for
#       each function type.


def CalculateGreedyMemoryAllocationOffset(memoryAllocations: list[MemoryAllocation]):
    # TODO: We are making a very simple algorithm right now. We can always improve this in the future if needed.
    #       In fact, it is preferable since if we find a problem the problematic input can help us find out how best to approach this.

    def GetFirstValidPointAfter(layerIndex, pointToStart):
        def Collision(point, range):
            if point >= range[0] and point < range[1]:
                return True
            return False

        layer = layers[layerIndex]
        currentPoint = pointToStart + 1
        for range in layer:
            if Collision(currentPoint, range):
                currentPoint = range[1]

        return currentPoint

    def FindCollision(layerIndex, point, size):
        layer = layers[layerIndex]
        for range in layer:
            if point < range[0] and point + size > range[0]:
                return range[0]
        return None

    def AddRegion(layerIndex, point, size):
        layer = layers[layerIndex]
        layer.append([point, point + size])

        # Slow
        layer = list(sorted(layer, key=lambda x: x[0]))

    if not memoryAllocations:
        return 0, []

    # Layers are just a list of ordered ranges. No point making a proper struct for such simple use case
    totalCycles = max([x.lastCycle for x in memoryAllocations])
    layers: list[list[int | int]] = [[] for x in range(totalCycles)]
    offsets: list[int] = [0] * (totalCycles - 1)
    totalMemoryNeeded = 0
    for index, memAlloc in enumerate(memoryAllocations):
        size = memAlloc.amount

        foundFit = False
        currentPoint = -1
        while not foundFit:
            currentPoint = GetFirstValidPointAfter(memAlloc.firstCycle, currentPoint)

            canFit = True
            for layer in range(memAlloc.firstCycle + 1, memAlloc.lastCycle):
                collisionPoint = FindCollision(layer, currentPoint, size)

                if collisionPoint:
                    bestValidPoint = GetFirstValidPointAfter(layer, collisionPoint)
                    canFit = False

            if canFit:
                foundFit = True
                offsets[index] = currentPoint
                totalMemoryNeeded = max(totalMemoryNeeded, currentPoint + size)
                for layer in range(memAlloc.firstCycle + 1, memAlloc.lastCycle):
                    AddRegion(layer, currentPoint, size)

    return totalMemoryNeeded, offsets


def IndexOfNodesThatUseOutput(cModel, outputName):
    indexes = []
    for index, op in enumerate(cModel.operations):
        for inp in op.inputs:
            if inp.name == outputName:
                indexes.append(index)
    return indexes


def CalculateMemoryAllocations(cModel):
    memoryAllocations = []
    for index, c in enumerate(cModel.operations):
        indexes = IndexOfNodesThatUseOutput(cModel, c.output)

        if not indexes:
            continue
        else:
            lastCycle = max(indexes)

        # In order to prevent operations that write on top of their input
        lastCycle += 1

        # Very simple memory calculation, might be wrong, especially if we decide to add padding and stuff like that. Need to make this more customizable.
        # TODO: We are also not handling the fact that some layers might support different tensor types.
        memoryRequired = 4  # Size of a float
        for dim in c.outputDimensions:
            memoryRequired *= TensorSize(dim)

        memoryAllocations.append(MemoryAllocation(index, lastCycle, memoryRequired))

    totalTempMemoryNeeded, offsets = CalculateGreedyMemoryAllocationOffset(
        memoryAllocations
    )

    # Embedded does not support unaligned memory. Need to be very carefully with all the allocations that are just passed directly to the embedded this way
    for x in offsets:
        assert x % 4 == 0

    cModel.tempMemoryNeeded = totalTempMemoryNeeded

    ptr = 0
    for index, c in enumerate(cModel.operations):
        indexes = IndexOfNodesThatUseOutput(cModel, c.output)

        if not indexes:
            continue

        c.outputMemoryAddress = MemoryLocation(offsets[ptr], MemoryType.TEMP)
        ptr += 1

    totalOutputMemory = 0
    outputOffsets = []
    for index, c in enumerate(cModel.operations):
        indexes = IndexOfNodesThatUseOutput(cModel, c.output)

        if indexes:
            continue

        # TODO: Support different tensor types and whatnot.
        memoryRequired = 4  # Size of a float
        for dim in c.outputDimensions:
            memoryRequired *= TensorSize(dim)

        outputOffsets.append(totalOutputMemory)
        c.outputMemoryAddress = MemoryLocation(totalOutputMemory, MemoryType.OUTPUT)
        totalOutputMemory += memoryRequired

    cModel.outputMemoryNeeded = totalOutputMemory
