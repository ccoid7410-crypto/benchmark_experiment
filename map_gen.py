"""Compatibility wrapper for the split map generator module."""

from maci_components.map_generator import MapGenerator


if __name__ == '__main__':
    map_gen = MapGenerator(23, 23)

    for _ in range(int(input("Number of maps to test: "))):
        map_gen.get_random_map()
        map_gen.display()
