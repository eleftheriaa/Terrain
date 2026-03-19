import heapq



def dijkstra(graph, start, goal):

    queue = [(0, start)]
    distances = {start: 0}
    came_from = {start: None}

    while queue:
        cost, current = heapq.heappop(queue)
        if current == goal:
            break

        for neighbor, weight in graph[current]:
            new_cost = cost + weight
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    if goal not in came_from:
        return [], float('inf')  # No path found


    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()

    return path, distances.get(goal, float("inf"))
