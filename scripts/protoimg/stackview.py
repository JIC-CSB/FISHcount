"""Stack viewer class. Allows viewing annotation."""

import os
import argparse

import scipy.misc
import numpy as np

import pygame
import pygame.image
from pygame.locals import *

def init_video(xdim, ydim):

    os.environ['SDL_VIDEO_CENTERED'] = '1'

    screen = pygame.display.set_mode((xdim, ydim), pygame.NOFRAME)

    return screen

def update_screen_2D(screen, im_surface, annot_surface=None, show_annot=False):

    screen.blit(im_surface, (0, 0))

    if show_annot:
        screen.blit(annot_surface, (0,0))

    pygame.display.flip()

def display_np_array_3D(im_array, annot_array=None):

    sf = 2

    xdim, ydim, zdim = im_array.shape

    if annot_array is not None:
        annot_array = np.transpose(annot_array, (1, 0, 2))

    screen = init_video(sf * xdim, sf * ydim)

    z_surfaces = []

    for z in range(zdim):
        z_plane = np.transpose(im_array[:,:,z])
        z_rgb_array = np.dstack([z_plane, z_plane, z_plane])
        z_surf = pygame.surfarray.make_surface(z_rgb_array)
        z_surf = pygame.transform.scale(z_surf, (2 * xdim, 2 * ydim))
        z_surfaces.append(z_surf)

    if annot_array is not None:
        annot_surface = pygame.surfarray.make_surface(annot_array)
        annot_surface = pygame.transform.scale(annot_surface, (2 * xdim, 2 * ydim))
        annot_surface.set_colorkey((0, 0, 0))
        show_annot = True
    else:
        annot_surface = None
        show_annot = False

    current_z = 0

    update_screen_2D(screen, z_surfaces[current_z], annot_surface, show_annot)
    # screen.blit(z_surfaces[current_z], (0, 0))
    # pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_a:
                    show_annot = not show_annot
                    update_screen_2D(screen, z_surfaces[current_z], annot_surface, show_annot)
            elif event.type == MOUSEBUTTONDOWN:
                b = event.button
                if b == 4:
                    current_z += 1
                if b == 5:
                    current_z -= 1

                current_z = max(0, current_z)
                current_z = min(zdim-1, current_z)

                update_screen_2D(screen, z_surfaces[current_z], annot_surface, show_annot)

    pygame.quit()


def display_np_array_2D(im_array, annot_array=None):

    im_array = np.transpose(im_array)

    xdim, ydim = im_array.shape
    
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    d_array = np.dstack([im_array, im_array, im_array])
    im_surface = pygame.surfarray.make_surface(d_array)

    if annot_array is not None:
        annot_surface = pygame.surfarray.make_surface(annot_array)
        annot_surface.set_colorkey((0, 0, 0))
        show_annot = True
    else:
        annot_surface = None
        show_annot = False

    im_surface = pygame.transform.scale(im_surface, (2 * xdim, 2 * ydim))
    screen = pygame.display.set_mode((2 * xdim, 2 * ydim), pygame.NOFRAME)

    update_screen_2D(screen, im_surface, annot_surface, show_annot)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_a:
                    show_annot = not show_annot
                    update_screen_2D(screen, im_surface, annot_surface, show_annot)
                if event.key == K_ESCAPE:
                    running = False
            elif event.type == MOUSEBUTTONDOWN:
                print event.button

    pygame.quit()

def display_image(filename):
    im_array = np.transpose(scipy.misc.imread(filename))

    display_np_array_2D(im_array)

def main():
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument('image_file', help="Name of image file.")

    args = parser.parse_args()

    display_image(args.image_file)


if __name__ == "__main__":
    main()
