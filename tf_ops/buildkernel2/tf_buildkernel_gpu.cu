#ifndef M_PI
#define M_PI           3.141592653589793F  /* pi */
#endif

#ifndef M_EPS
#define M_EPS          1.01e-3F             /* epsilon */
#endif
#include "svd3_cuda.h"
struct point3d
{
    float x=0, y=0, z=0;
};

// database:  B*N*3, (x,y,z)
// query:     B*M*3, (x,y,z)
// nnIndex:   B*M*K
// nnCount:   B*M
// nnDist:    B*M*K
// filtIndex: B*M*K
// rotateXyz: B*M*K*3
__global__ void build_spherical_kernel(const int B, const int N, const int M, const int K,
                                             const int n, const int p, const int q, const float radius,
                                             const float* database, const float* query, const int* nnIndex,
                                             const int* nnCount, const float* nnDist, int* filtIndex,float* rotateXyz)
{
    // get the neighbor indices
    point3d ptQuery, pt, delta;
    for(int i=blockIdx.x;i<B;i+=gridDim.x) //for batch
    {
        for(int j=threadIdx.x;j<M;j+=blockDim.x) //for query point
        {
            int qf = i*M*3+j*3;//query ponit offset of query
            ptQuery.x = query[qf];//query shape is B*M*3
            ptQuery.y = query[qf+1];
            ptQuery.z = query[qf+2];

            int nnSize = nnCount[i*M+j];

            int bf = i*N*3; //batch offset of database
            int bfi = i*M*K+j*K;//batch and point offset of nnIndex

            int rf = i*M*K*3 + j*K*3;//batch and point offset of rotateXyz

            bool transbool = true;
            // transbool = false;
            float trans[9] = {0};
            if(transbool)
            {
                float mean[3] = {0,0,0};
                // mean(x)
                for(int k=0;k<nnSize;k++)
                {
                    int ptID = nnIndex[bfi+k];   // input point ID                
                    mean[0] += database[bf+ptID*3];
                    mean[1] += database[bf+ptID*3+1];
                    mean[2] += database[bf+ptID*3+2];
                }
                for(int i=0;i<3;i++)
                    mean[i] = mean[i]/nnSize;

                //cov = (x-mean)'*(x-mean)/nnSize
                float cov[9]={0};
                for(int i = 0;i<3;i++)
                {
                    for(int j = i; j <3;j++)
                    {
                        for(int k = 0;k < nnSize;k++)    //database[3*k:3*k+2] a point
                        {
                            int ptID = nnIndex[bfi+k];
                            // cov[i*3+j] += (database[bf+ptID*3+i]-mean[i])*(database[bf+ptID*3+j]-mean[j]);
                            cov[i*3+j] += (database[bf+ptID*3+i]-query[qf+i])*(database[bf+ptID*3+j]-query[qf+j]);
                        }
                        cov[i+j*3] = cov[i*3+j];
                    }
                }
                for(int i =0;i<9;i++)
                    cov[i] = cov[i]/nnSize;


                int Id = 0;
                float maxdis = 0;
                for(int k=0;k<nnSize;k++)
                {
                    if(nnDist[bfi+k]>maxdis)
                     {
                        maxdis = nnDist[bfi+k];
                        Id = k;
                     }   
                }
                Id = nnIndex[bfi+Id];
                for(int i=0;i<3;i++)
                    mean[i] = database[bf+Id*3+i]- query[qf+i];

                // for(int i=0;i<3;i++)
                //     mean[i] = mean[i]- query[qf+i];
                //calc Rotate Mat    
                // calcRotateMat(trans,&query[qf],cov);
                calcRotateMat(trans,&mean[0],cov);
            }
            //find bins with trans

            for(int k=0;k<nnSize;k++)
            {
                int ptID = nnIndex[bfi+k];   // input point ID?

                float deltaxyz[3] = {0};
                pt.x = database[bf+ptID*3];      // the neighbor points
                pt.y = database[bf+ptID*3+1];
                pt.z = database[bf+ptID*3+2];

                delta.x = pt.x - ptQuery.x;
                delta.y = pt.y - ptQuery.y;
                delta.z = pt.z - ptQuery.z;
                
                if(transbool){
                    //delta = delta*trans
                    for(int r = 0;r<3;r++)
                        deltaxyz[r] = delta.x*trans[r]+delta.y*trans[r+3]+delta.z*trans[r+6];
                    delta.x = deltaxyz[0];
                    delta.y = deltaxyz[1];
                    delta.z = deltaxyz[2];
                }
                rotateXyz[rf+3*k+0] = delta.x;
                rotateXyz[rf+3*k+1] = delta.y;
                rotateXyz[rf+3*k+2] = delta.z;
                //find bins
                float dist = nnDist[bfi+k];
                float dist2D = delta.x*delta.x + delta.y*delta.y;
                dist2D = sqrtf(dist2D);

                filtIndex[bfi+k] = 0;
                if (dist>M_EPS && fabs(dist-M_EPS)>1e-6) // update the bin index
                {
                    float theta = atan2f(delta.y, delta.x);
                    float phi = atan2f(delta.z, dist2D);

                    theta = theta<M_PI?theta:(-M_PI);
                    theta = theta>(-M_PI)?theta:(-M_PI);
                    theta += M_PI;

                    phi = phi<(M_PI/2)?phi:(M_PI/2);
                    phi = phi>(-M_PI/2)?phi:(-M_PI/2);
                    phi += M_PI/2;

                    float alpha = theta*n/2/M_PI;
                    float beta = phi*p/M_PI;
                    float gamma = dist*q/(radius+1e-6F);

                    int nID = min(n-1, int(alpha));
                    int pID = min(p-1, int(beta));
                    int qID = min(q-1, int(gamma));

                    filtIndex[bfi+k] = qID*p*n + pID*n + nID + 1;
                }
            }
        }
    }
}



void sphericalKernelLauncher(int B, int N, int M, int K, int n, int p, int q, float radius,
                                  const float* database, const float* query, const int* nnIndex,
                                  const int* nnCount, const float* nnDist, int* filtIndex,float* rotateXyz)
{
    build_spherical_kernel<<<32,1024>>>(B, N, M, K, n, p, q, radius,
                                database, query, nnIndex, nnCount, nnDist, filtIndex,rotateXyz);
}

