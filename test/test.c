#include <stdio.h>
#include <stdlib.h>
#define TAILLE_MAX 100
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int main()
{
    int mat[3][2] = {{1, 2}, {3, 4}, {5, 6}};

    printf("%d\n", mat[1][1]);

    for (int i = 0; i < -1; i++)
    {
        printf("Hello");
    }

    // int r;
    // r = MAX(5, 6);
    // printf("%d\n", r);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

int main()
{
    printf("Image Processing!\n");

    Image *image = parse_image("./data/image.pgm");

    Coordinates coords = {18, 50};

    print_image(image, &coords, 20, 20);

    printf("pixel at 18:53 %d \n", image->image[18][53]);

    Hist *hist = make_hist(image, 0);

    printf("number of pixels at 193: %.2f \n", hist->hist[193]);

    // printf("number of pixels at 193: %d \n", hist->hist_coordinates[193].length);
    printf("coordinates of the first pixel of shade 193: x: %d, y: %d \n", hist->hist_coordinates[193].coordinates[0].x, hist->hist_coordinates[193].coordinates[0].y);

    printf("luminousity: %.2f \n", luminousity(image));

    printf("contrast: %.2f \n", contrast(image));

    // plot_hist(hist);

    print_hist(hist);

    return 0;
}
//
//
// int main(int argc, char *argv[])
// {
//     FILE* fichier = NULL;
//     char n[TAILLE_MAX] = "" , pixMax[5] = "";
//     int i,  j, b,  dimension[3] = {0};
//     char ligne, colonne ;
//     char caractereActuel;
//     fichier = fopen("image.PGM", "r");
//     if (fichier != NULL)
//     {
//         i = 0;
//         j = 0;
//
//         while (fgetc(fichier)!= EOF && i != 4) {
//
//             // fseek(fichier, -1, SEEK_CUR);
//
//             if (j == 0) {
//                 // printf("%d\n", j );
//                 while (j != 2) {
//                     caractereActuel = fgetc(fichier);
//                     if (caractereActuel == '\n') {
//                         j = j + 1;
//                     }
//                 }
//             }
//
//             if (i == 0) {
//                 fscanf(fichier, "%d %d\n", &dimension[0], &dimension[1]);
//
// 				i = i + 1;
//
//
//             }
//
// 			if (i == 1) {
// 				fgets(pixMax, 5, fichier);
// 			}
//
//
//             while (fgets(n, TAILLE_MAX, fichier) != NULL && i < 4) // On lit le fichier tant qu'on ne reçoit pas d'erreur (NULL)
//             {
//
//                 printf("%s\n", n );
//                 i = i + 1;
//             }
//
//
//
//         }
//
//         printf("ligne = %d\n", dimension[0]);
//         printf("colonne = %d\n", dimension[1]);
//         printf("pixMax = %s\n", pixMax );
//
//         fclose(fichier);
//     }
//
//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "filters.h"

int main()
{
    Coordinates coords = {0, 0};
    Image *feep = parse_image("./data/feep.ascii.pgm");
    Image *mona_lisa = parse_image("./data/mona_lisa.ascii.pgm");
    Image *apollo = parse_image("./data/apollonian_gasket.ascii.pgm");
    SpatialFilter *moyenneur = create_filters();

    // play with fleep
    Image *feep_moyen = convulv(feep, moyenneur, MIRROR_EDGES);
    // print_image(feep_moyen, &coords, feep->height - 1, feep->width - 1);
    save(feep_moyen, "./data/feep_moyen_3.pgm");

    return 0;
}
