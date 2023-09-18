//
//  mean_square_displacement.c
//
//
//  Created by Cibrán López Álvrez on 25/11/2021.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdbool.h>


void ExitError(const char *miss, int errcode) {
    fprintf(stderr, "\nERROR: %s.\nStopping...\n\n", miss);
    exit(errcode);
}

void particle_i(int type, int ncon, int np, int N_components, int np_i[N_components], int ntimes, long double scale, int nconcort, int nposcor, long double r_x[np*ncon], long double r_y[np*ncon], long double r_z[np*ncon], long double rperf_x[np], long double rperf_y[np], long double rperf_z[np], long double rperfc_x[np], long double rperfc_y[np], long double rperfc_z[np], long double rdp[ncon], long double rd2p[ncon], long double rc_x[np*ncon], long double rc_y[np*ncon], long double rc_z[np*ncon], long double bc[3][3], long double rposcor_x[ncon], long double rposcor_y[ncon], long double rposcor_z[ncon]) {
    
    int i, j, k, np0 = 0;
    char type_str[5], msd_corr_file[30] = "msd_corr_", msd_file[30] = "msd_";
    
    sprintf(type_str, "%u", type);
    strcat(msd_corr_file, type_str);
    strcat(msd_corr_file, ".dat");
    strcat(msd_file, type_str);
    strcat(msd_file, ".dat");
    
    for (i = 0; i < type; i++)
        np0 += np_i[i];
    
    int npf = np0 + np_i[type];
    
    printf("Type %u with np0 = %u and npf = %u\n", type, np0, npf);
    for (i = 0; i < ncon; i++) {
        rdp[i] = 0;
        rd2p[i] = 0;
    }
    
    for (k = 0; k < ntimes; k++)
        for (j = k; j < nconcort+k; j++)
            for (i = np0; i < npf; i++) {
                long double r2 = powl(rc_x[i*ncon + j] - rc_x[i*ncon + k], 2) + powl(rc_y[i*ncon + j] - rc_y[i*ncon + k], 2) + powl(rc_z[i*ncon + j] - rc_z[i*ncon + k], 2);
                
                rdp[j-k] += r2;
                rd2p[j-k] += r2 * r2;
            }
    
    long double aux = 1.0 / (float) ((npf - np0) * ntimes);
    for (i = 0; i < nconcort; i++) {
        rdp[i] *= aux;
        rd2p[i] *= aux;
    }
    
    for (i = 0; i < ncon; i++) {
        rposcor_x[i] = 0;
        rposcor_y[i] = 0;
        rposcor_z[i] = 0;
    }
    
    for (k = 0; k < (ncon-nposcor); k++) {
        for (j = k; j < nposcor+k; j++) {
            for (i = np0; i < npf; i++) {
                rposcor_x[j-k] += (rc_x[i*ncon + j] - rperfc_x[i]) * (rc_x[i*ncon + k] - rperfc_x[i]);
                rposcor_y[j-k] += (rc_y[i*ncon + j] - rperfc_y[i]) * (rc_y[i*ncon + k] - rperfc_y[i]);
                rposcor_z[j-k] += (rc_z[i*ncon + j] - rperfc_z[i]) * (rc_z[i*ncon + k] - rperfc_z[i]);
            }
        }
        //printf("%u %lf %lf\n", k, rc_x[k], rperfc_x[0]);
    }
    
    FILE *msd_corr;
    if ((msd_corr = fopen(msd_corr_file, "w")) == NULL)
        ExitError("The msd_corr data file cannot be opened", 9);
    
    aux = scale * scale / ((float) ((npf - np0) * (ncon-nposcor)));
    for (i = 0; i < nposcor; i++) {
        rposcor_x[i] *= aux;
        rposcor_y[i] *= aux;
        rposcor_z[i] *= aux;
        
        fprintf(msd_corr, "%u %.19Lf\n", i, rposcor_x[i] + rposcor_y[i] + rposcor_z[i]);
    } fclose(msd_corr);
    
    FILE *msd;
    if ((msd = fopen(msd_file, "w")) == NULL)
        ExitError("The msd data file cannot be opened", 10);
    
    aux = scale * scale;
    for (i = 0; i < nconcort; i++) {
        fprintf(msd, "%u %.19Lf %.19Lf\n", i, (rdp[i] * aux), sqrtl((rd2p[i] - (rdp[i] * rdp[i])) / ((float) (ntimes-1))) * aux);
    } fclose(msd);
}

bool valid_str(const char* str) {
    // Checks whether the token is valid for counting the number of elements in the compound
    size_t len = strlen(str);
    for (size_t i = 0; i < len; i++) {
        if (str[i] != '\n') {
            return true;
        }
    }
    return false;
}

/* THIS PROGRAM CALCULATES THE MEAN SQUARED DISPLACEMENT OF PARTICLES AND THE POSITION CORRELATION FUNCTION FROM XDATCAR FILE GENERATED WITH md.x (it corrects periodic boundary conditions) */
int main() {
    int i, j, k, ncon = 0;
    char *POSCAR_file_name = "POSCAR", *XDATCAR_file_name = "XDATCAR";
    char *token, *data_string = NULL;
    
    size_t buffer_size = 0;
    
    // ----------------------------------
    // Calculate number of configurations
    // ----------------------------------
    
    FILE *XDATCAR;
    if ((XDATCAR = fopen(XDATCAR_file_name, "r")) == NULL)
        ExitError("XDATCAR data file cannot be opened", 1);
    
    int np = 0, *np_i, N_components = 0;
    
    while (getline(&data_string, &buffer_size, XDATCAR) != EOF) {
        ncon += 1;
        
        if (ncon == 6) {
            token = strtok(data_string, " ");
            
            while (token != NULL){
                if (valid_str(token)) // It detected a token with valid elements
                    N_components += 1;

                token = strtok(NULL, " ");
            }
            
            if (((np_i  = (int*) malloc(N_components * sizeof(int)))) == NULL)
                ExitError("When allocating memory for np", 2);
        }
        
        if (ncon == 7) { // Reading number of particles.
            token = strtok(data_string, " ");
            
            for (i = 0; i < N_components; i++) {
                np_i[i] = atoi(token);
                np += atoi(token);
                token = strtok(NULL, " ");
            }
        }
    }
    
    printf("Components: N_c = %u, ", N_components);
    for (i = 0; i < N_components; i++)
        printf("n_%u = %u, ", i, np_i[i]);
    printf("N_T = %u\n", np);
    
    ncon = (ncon - 6) / ((float) (np+1));
    printf("NPT: %u\n", np);
    printf("NCON: %u\n", ncon);
    
    int nconcort = 0.5 * ncon, nposcor = 0.5 * ncon, ntimes = ncon - nconcort;
    long double rperf_x[np], rperf_y[np], rperf_z[np], rperfc_x[np], rperfc_y[np], rperfc_z[np], rdp[ncon], rd2p[ncon], bc[3][3], rposcor_x[ncon], rposcor_y[ncon], rposcor_z[ncon];
    long double *r_x, *r_y, *r_z, *rc_x, *rc_y, *rc_z;
    
    if (((r_x  = (long double*) malloc(np*ncon * sizeof(long double)))) == NULL)
        ExitError("When allocating memory for r_x", 2);
    
    if (((r_y  = (long double*) malloc(np*ncon * sizeof(long double)))) == NULL)
        ExitError("When allocating memory for r_y", 3);
    
    if (((r_z  = (long double*) malloc(np*ncon * sizeof(long double)))) == NULL)
        ExitError("When allocating memory for r_z", 4);

    if (((rc_x  = (long double*) malloc(np*ncon * sizeof(long double)))) == NULL)
        ExitError("When allocating memory for rc_x", 5);

    if (((rc_y  = (long double*) malloc(np*ncon * sizeof(long double)))) == NULL)
        ExitError("When allocating memory for rc_y", 6);

    if (((rc_z  = (long double*) malloc(np*ncon * sizeof(long double)))) == NULL)
        ExitError("When allocating memory for rc_z", 7);
    
    // -------------------------------------
    // Read configurations from XDATCAR file
    // -------------------------------------
    
    rewind(XDATCAR);
    
    for (i = 0; i < 7; i++) // For my specific case.
        getline(&data_string, &buffer_size, XDATCAR);
    
    for (i = 0; i < ncon; i++) {
        getline(&data_string, &buffer_size, XDATCAR);
        for (j = 0; j < np; j++) {
            getline(&data_string, &buffer_size, XDATCAR);
            token = strtok(data_string, " ");
            r_x[j*ncon + i] = atof(token);
            
            token = strtok(NULL, " ");
            r_y[j*ncon + i] = atof(token);
            
            token = strtok(NULL, " ");
            r_z[j*ncon + i] = atof(token);
        }
    }
    
    // -------------------------------------
    // Read configurations from POSCAR file
    // -------------------------------------
    
    FILE *POSCAR;
    if ((POSCAR = fopen(POSCAR_file_name, "r")) == NULL)
        ExitError("POSCAR data file cannot be opened", 8);
    
    getline(&data_string, &buffer_size, POSCAR);
    getline(&data_string, &buffer_size, POSCAR);
    long double scale = atof(strtok(data_string, " "));
    printf("Scale: %Lf\n", scale);
    
    printf("\nBASE\n");
    for (i = 0; i < 3; i++) {
        getline(&data_string, &buffer_size, POSCAR);
        token = strtok(data_string, " ");
        
        for (j = 0; j < 3; j++) {
            bc[j][i] = atof(token);
            token = strtok(NULL, " ");
            printf("%Lf\n", bc[j][i]);
        }
    }
    
    getline(&data_string, &buffer_size, POSCAR);
    getline(&data_string, &buffer_size, POSCAR);
    
    for (i = 0; i < np; i++) {
        getline(&data_string, &buffer_size, POSCAR);
        token = strtok(data_string, " ");
        
        if (isalpha(token[0])) {
            i -= 1;
            continue;
        }
        
        rperf_x[i] = atof(token);
        
        token = strtok(NULL, " ");
        rperf_y[i] = atof(token);
        
        token = strtok(NULL, " ");
        rperf_z[i] = atof(token);
        
        r_x[i*ncon] = rperf_x[i];
        r_y[i*ncon] = rperf_y[i];
        r_z[i*ncon] = rperf_z[i];
    }
    
    // ---------------------------------------------------------
    // Correct positions because of periodic boundary conditions
    // ---------------------------------------------------------
    
    long double rx_aux, ry_aux, rz_aux;
        
    for (j = 0; j < ncon-1; j++)
        for (i = 0; i < np; i++) {
            rx_aux = r_x[i*ncon + j+1] - r_x[i*ncon + j];
            ry_aux = r_y[i*ncon + j+1] - r_y[i*ncon + j];
            rz_aux = r_z[i*ncon + j+1] - r_z[i*ncon + j];
            
            if (fabsl(rx_aux) > 0.5) {
                if (rx_aux < 0)
                    for (k = j+1; k < ncon; k++)
                        r_x[i*ncon + k] += 1.0;
                else
                    for (k = j+1; k < ncon; k++)
                        r_x[i*ncon + k] -= 1.0;
            }
            
            if (fabsl(ry_aux) > 0.5) {
                if (ry_aux < 0)
                    for (k = j+1; k < ncon; k++)
                        r_y[i*ncon + k] += 1.0;
                else
                    for (k = j+1; k < ncon; k++)
                        r_y[i*ncon + k] -= 1.0;
            }
            
            if (fabsl(rz_aux) > 0.5) {
                if (rz_aux < 0)
                    for (k = j+1; k < ncon; k++)
                        r_z[i*ncon + k] += 1.0;
                else
                    for (k = j+1; k < ncon; k++)
                        r_z[i*ncon + k] -= 1.0;
            }
        }
    
    // --------------------------------
    // Convert to cartesian coordinates
    // --------------------------------
    
    for (j = 0; j < ncon; j++)
        for (i = 0; i < np; i++) {
            rc_x[i*ncon + j] = (r_x[i*ncon + j] * bc[0][0]) + (r_y[i*ncon + j] * bc[0][1]) + (r_z[i*ncon + j] * bc[0][2]);
            rc_y[i*ncon + j] = (r_x[i*ncon + j] * bc[1][0]) + (r_y[i*ncon + j] * bc[1][1]) + (r_z[i*ncon + j] * bc[1][2]);
            rc_z[i*ncon + j] = (r_x[i*ncon + j] * bc[2][0]) + (r_y[i*ncon + j] * bc[2][1]) + (r_z[i*ncon + j] * bc[2][2]);
        }
    
    for (i = 0; i < np; i++) {
        rperfc_x[i] = (rperf_x[i] * bc[0][0]) + (rperf_y[i] * bc[0][1]) + (rperf_z[i] * bc[0][2]);
        rperfc_y[i] = (rperf_x[i] * bc[1][0]) + (rperf_y[i] * bc[1][1]) + (rperf_z[i] * bc[1][2]);
        rperfc_z[i] = (rperf_x[i] * bc[2][0]) + (rperf_y[i] * bc[2][1]) + (rperf_z[i] * bc[2][2]);
    }
    
    // -----------------------------------------------------------
    // Computation of the mean squared displacement once positions
    // are corrected for the periodic boundary conditions
    // -----------------------------------------------------------
    
    for (i = 0; i < N_components; i++)
        particle_i(i, ncon, np, N_components, np_i, ntimes, scale, nconcort, nposcor, r_x, r_y, r_z, rperf_x, rperf_y, rperf_z, rperfc_x, rperfc_y, rperfc_z, rdp, rd2p, rc_x, rc_y, rc_z, bc, rposcor_x, rposcor_y, rposcor_z);
}
