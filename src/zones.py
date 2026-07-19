def get_zone(x, y, width, height):

    # Exit
    if x > width * 0.85:
        return "EXIT"

    # Checkout
    if x < width * 0.25 and y > height * 0.65:
        return "CHECKOUT"

    # Shelf Left
    if x < width * 0.35:
        return "SHELF_A"

    # Shelf Center
    if x < width * 0.70:
        return "SHELF_B"

    return "SHELF_C"