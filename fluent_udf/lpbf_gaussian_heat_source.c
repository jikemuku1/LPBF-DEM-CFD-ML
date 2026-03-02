/*=============================================================================
 * ANSYS Fluent UDF — LPBF Conical Gaussian Laser Heat Source
 *
 * Description : Implements a conical Gaussian laser heat source for
 *               Laser Powder Bed Fusion (LPBF) simulation, including:
 *                 - Gaussian beam irradiance with conical beam profile
 *                 - Convective and radiative surface heat losses
 *                 - VOF-gradient-based surface normal computation
 *                 - Recoil pressure source terms (x, y, z) via
 *                   Hertz-Knudsen evaporation model
 *
 * Material    : 18Ni300 maraging steel
 * Solver      : ANSYS Fluent 2022 R2, VOF multiphase, pressure-based
 * Compilation : Load via Fluent GUI:
 *               Define > User-Defined > Functions > Compiled
 *               (add this .c file as source, build, and load)
 *
 * UDM slots required (set in Fluent before loading):
 *   UDM-0 : dAlpha/dx
 *   UDM-1 : dAlpha/dy
 *   UDM-2 : dAlpha/dz
 *   UDM-3 : |grad(Alpha)|  (magnitude)
 *
 * Hook assignments in Fluent:
 *   ADJUST   hook  -> adjust_gradient
 *   Energy   source -> heat_source    (metal phase)
 *   X-mom.   source -> x_pressure     (mixture)
 *   Y-mom.   source -> y_pressure     (mixture)
 *   Z-mom.   source -> z_pressure     (mixture)
 *
 * Reference   : https://github.com/jikemuku1/LPBF-DEM-CFD-ML
 * License     : MIT
 *=============================================================================*/

#include "udf.h"
#include "sg_mphase.h"
#include "mem.h"
#include "sg_mem.h"
#include "math.h"
#include "flow.h"
#include "unsteady.h"
#include "metric.h"

/* ── Laser process parameters ─────────────────────────────────────────────── */
#define A    0.4          /* laser absorptivity (dimensionless)                */
#define P    280          /* laser power (W)                                   */
#define R    324e-7       /* beam radius at the powder-bed surface (m)         */
#define v    0.95         /* laser scan speed (m s-1)                          */

/* ── Thermal boundary condition parameters ────────────────────────────────── */
#define h    25           /* convective heat transfer coefficient (W m-2 K-1)  */
#define Ta   298          /* ambient / initial temperature (K)                 */
#define s    5.67e-8      /* Stefan-Boltzmann constant (W m-2 K-4)             */
#define e    0.5          /* surface emissivity of 18Ni300 (dimensionless)     */

/* ── Mathematical constant ────────────────────────────────────────────────── */
#define Pi   3.1415926535

/* ── Phase-transition temperatures for 18Ni300 maraging steel ─────────────── */
#define Ts   1387         /* solidus temperature (K)                           */
#define Tl   1441         /* liquidus temperature (K)                          */

/* ── Initial laser beam centre position ───────────────────────────────────── */
#define x0   -342e-6      /* initial x-coordinate of beam centre (m)          */
#define y0   0.0          /* y-coordinate of beam centre (m)                  */
#define z0   0.0          /* z-coordinate of beam centre (m, unused)          */

/* ── Evaporation / recoil-pressure model constants ────────────────────────── */
#define Lv   6.09e6       /* latent heat of vaporisation for 18Ni300 (J kg-1) */
#define Tv   2862         /* boiling point temperature (K)                     */
#define Rg   8.314        /* universal gas constant (J mol-1 K-1)              */
#define M    0.05843      /* molar mass of 18Ni300 (kg mol-1)                  */
#define Pa   101325       /* reference atmospheric pressure (Pa)               */

/* ── Infrastructure macros (declared but not used in active physics) ───────── */
#define domain_ID 10
#define current   40
#define voltage   60

/* ── Global: conical beam entry/exit depths (set per-call in heat_source) ─── */
real De, Ds;

/*=============================================================================
 * DEFINE_ADJUST  —  compute VOF gradient and store in UDM 0–3
 *
 * Called once per iteration BEFORE source-term evaluation.
 * Stores the gradient of the secondary-phase (metal) VOF in UDMs so that
 * the recoil-pressure UDFs can use the surface normal without an extra
 * Fluent gradient call inside the source macro.
 *===========================================================================*/
DEFINE_ADJUST(adjust_gradient, domain)
{
#if !RP_HOST
    Thread *t;
    Thread **pt;
    cell_t  c;
    int phase_domain_index = 1;   /* 0 = gas (primary), 1 = metal (secondary) */

    Domain *pDomain = DOMAIN_SUB_DOMAIN(domain, phase_domain_index);

    /* Allocate gradient reconstruction buffers */
    Alloc_Storage_Vars(pDomain, SV_VOF_RG, SV_VOF_G, SV_NULL);
    Scalar_Reconstruction(pDomain, SV_VOF, -1, SV_VOF_RG, NULL);
    Scalar_Derivatives(pDomain, SV_VOF, -1, SV_VOF_G, SV_VOF_RG,
                       Vof_Deriv_Accumulate);

    mp_thread_loop_c(t, domain, pt)
    {
        if (FLUID_THREAD_P(t))
        {
            Thread *ppt = pt[phase_domain_index];
            begin_c_loop(c, t)
            {
                C_UDMI(c, t, 0) = C_VOF_G(c, ppt)[0];          /* dAlpha/dx  */
                C_UDMI(c, t, 1) = C_VOF_G(c, ppt)[1];          /* dAlpha/dy  */
                C_UDMI(c, t, 2) = C_VOF_G(c, ppt)[2];          /* dAlpha/dz  */
                C_UDMI(c, t, 3) = sqrt(                          /* |grad(a)| */
                    C_UDMI(c, t, 0) * C_UDMI(c, t, 0) +
                    C_UDMI(c, t, 1) * C_UDMI(c, t, 1) +
                    C_UDMI(c, t, 2) * C_UDMI(c, t, 2));
            }
            end_c_loop(c, t)
        }
    }

    Free_Storage_Vars(pDomain, SV_VOF_RG, SV_VOF_G, SV_NULL);
    Message0("adjust_gradient called\n");
#endif
}

/*=============================================================================
 * DEFINE_SOURCE  (heat_source)
 *
 * Net volumetric energy source (W m-3) at the metal-phase free surface:
 *
 *   q_net = q_laser - q_conv - q_rad
 *
 *   q_laser  = conical Gaussian irradiance (normalised over cone volume)
 *   q_conv   = h * (T - Ta)                  [convective loss]
 *   q_rad    = sigma * eps * (T^4 - Ta^4)    [radiative loss]
 *
 * Active only in interfacial cells: 0.05 < alpha_metal < 1.
 * The same formula applies below and above the boiling point (Tv);
 * above Tv the evaporative flux is handled by the recoil-pressure UDFs.
 * A density-ratio correction (rho / rho_inter) maps the mixed-cell energy
 * to the actual metal-cell contribution required by Fluent.
 *===========================================================================*/
DEFINE_SOURCE(heat_source, c, t, dS, eqn)
{
    Thread *pri_th = THREAD_SUB_THREAD(t, 0);  /* gas phase   */
    Thread *sec_th = THREAD_SUB_THREAD(t, 1);  /* metal phase */

    real source;
    real x[ND_ND];
    real time = RP_Get_Real("flow-time");      /* current time (s)            */
    C_CENTROID(x, c, t);

    real T     = C_T(c, t);
    real alpha = C_VOF(c, pri_th);            /* gas volume fraction         */
    real gamma = C_LIQF(c, sec_th);           /* liquid fraction in metal    */

    /* Vapour pressure (Clausius-Clapeyron) and evaporative mass flux */
    real Pv = 0.54 * Pa * exp((Lv * M * (T - Tv)) / (Rg * T * Tv));
    real mv = (0.82 * M * Pv) / (sqrt(2 * Pi * M * Rg * T));  /* kg m-2 s-1 */

    /* Temperature-dependent densities of 18Ni300 */
    real rhog  = 1.6228;                               /* gas density (kg m-3) */
    real rhos  = 8100 - 0.431 * T;                    /* solid density        */
    real rhol  = 7573 + 0.0755 * T - 0.00019 * pow(T, 2); /* liquid density   */
    real rhom  = rhol * gamma + rhos * (1 - gamma);   /* metal mixture density */
    real rho   = alpha * rhog + rhom * (1 - alpha);   /* cell mixture density  */

    /* Temperature-dependent specific heats */
    real Cpg   = 520.64;
    real Cps   = 491 + 0.0525 * T;
    real Cpl   = 725;
    real Cpm   = Cpl * gamma + Cps * (1 - gamma);
    real Cp    = (1 - alpha) * Cpm + Cpg * alpha;

    /* Density-weighting factor (computed for reference; not directly used below) */
    real factor = (2 * rho * Cp) / (rhom * Cpm + rhog * Cpg);

    real dens      = C_R(c, t);
    real den_inter = 0.5 * (C_R(c, pri_th) + C_R(c, sec_th));

    /* Track-offset variables — set to zero for single-track simulation;
       extend for multi-track by providing non-zero i (x-offset) or j (y-offset) */
    real i = 0;
    real j = 0;

    /* Conical beam geometry: beam narrows linearly from surface (Ds) to tip (De) */
    Ds = 0.00011;        /* z-level of powder-bed top surface (m)  */
    De = Ds - 0.00001;   /* z-level of cone tip (m)                */

    /* Only apply heat source at the liquid-gas interface */
    if (C_VOF(c, sec_th) > 0.05 && C_VOF(c, sec_th) < 1)
    {
        if (C_T(c, sec_th) < 2862)  /* below boiling point */
        {
            source = (((2 * A * P) / (Pi) / ((Ds - De) * (R * R + R * R / 2 + R * R / 4)))
                      * exp(-2 * (pow(x[1] - 0 - j, 2.0)
                                + pow(x[0] + 0.000342 - time * v - i, 2.0))
                            / (pow(R + (R - R / 2) / 0.00009 * (x[2] - Ds), 2.0)))
                      - h * (T - Ta)                        /* convective loss */
                      - s * e * (pow(T, 4) - pow(Ta, 4)))  /* radiative loss  */
                     * dens / den_inter;
            dS[eqn] = 0.0;
        }
        else  /* above boiling point — evaporation handled by recoil-pressure UDFs */
        {
            source = (((2 * A * P) / (Pi) / ((Ds - De) * (R * R + R * R / 2 + R * R / 4)))
                      * exp(-2 * (pow(x[1] - 0 - j, 2.0)
                                + pow(x[0] + 0.000342 - time * v - i, 2.0))
                            / (pow(R + (R - R / 2) / 0.00009 * (x[2] - Ds), 2.0)))
                      - h * (T - Ta)
                      - s * e * (pow(T, 4) - pow(Ta, 4)))
                     * dens / den_inter;
            dS[eqn] = 0.0;
        }
    }
    else
    {
        source  = 0.0;  /* no source outside the free-surface region */
        dS[eqn] = 0.0;
    }
    return source;
}

/*=============================================================================
 * DEFINE_SOURCE  (x_pressure / y_pressure / z_pressure)
 *
 * Recoil pressure momentum source terms along each coordinate axis.
 *
 * Physical basis: Above the liquidus (Tl = 1441 K) the evaporating metal
 * exerts a reaction (recoil) pressure on the melt surface. The magnitude
 * follows a Clausius-Clapeyron approximation:
 *
 *   P_recoil ~ ((1 + beta) / 2) * P_atm * exp(Lv*M*(T-Tv)/(Rg*T*Tv))
 *
 * where beta = 0.08 accounts for partial back-flux of vapour (Anisimov 1968).
 * The source is projected onto the surface normal via the VOF gradient
 * components stored in UDMs 0-2 by DEFINE_ADJUST.
 *
 * `state` is a smooth Heaviside function that switches on above Tl.
 *===========================================================================*/
DEFINE_SOURCE(x_pressure, c, t, dS, eqn)
{
#if !RP_HOST
    Thread *ta, *ts;
    real dens, vof_gx, vof_gy, vof_gz, vof_g, dens_inter, disc, source, vofc, F,
         temp, state;
    real xc[ND_ND];
    real flow_time, torch_loc;

    flow_time = RP_Get_Real("flow-time");
    C_CENTROID(xc, c, t);
    ta = THREAD_SUB_THREAD(t, 0);   /* gas phase   */
    ts = THREAD_SUB_THREAD(t, 1);   /* metal phase */

    vofc       = C_VOF(c, ts);
    dens       = C_R(c, t);
    temp       = C_T(c, t);
    dens_inter = 0.5 * (C_R(c, ta) + C_R(c, ts));

    /* Smooth step: 0 below liquidus, approaches 1 above */
    state = ((temp - 1441) + abs(temp - 1441)) / (2 * abs(temp - 1441) + 1);

    /* Recoil pressure x-component: P_recoil * n_x,  n_x = UDM-0 */
    source = ((1 + 0.08) / 2) * 101325
             * exp(((temp - 2862) / (8.314472 * temp * 2862)) * 796600)
             * C_UDMI(c, t, 0) * dens / dens_inter;

    source = state * source;
    return source;
#endif
}

DEFINE_SOURCE(y_pressure, c, t, dS, eqn)
{
#if !RP_HOST
    Thread *ta, *ts;
    real dens, vof_gx, vof_gy, vof_gz, vof_g, dens_inter, disc, source, vofc, F,
         temp, state;
    real xc[ND_ND];
    real flow_time, torch_loc;

    flow_time = RP_Get_Real("flow-time");
    C_CENTROID(xc, c, t);
    ta = THREAD_SUB_THREAD(t, 0);
    ts = THREAD_SUB_THREAD(t, 1);

    vofc       = C_VOF(c, ts);
    dens       = C_R(c, t);
    temp       = C_T(c, t);
    dens_inter = 0.5 * (C_R(c, ta) + C_R(c, ts));

    state = ((temp - 1441) + abs(temp - 1441)) / (2 * abs(temp - 1441) + 1);

    /* Recoil pressure y-component: P_recoil * n_y,  n_y = UDM-1 */
    source = ((1 + 0.08) / 2) * 101325
             * exp(((temp - 2862) / (8.314472 * temp * 2862)) * 796600)
             * C_UDMI(c, t, 1) * dens / dens_inter;

    source = state * source;
    dS[eqn] = 0;
    return source;
#endif
}

DEFINE_SOURCE(z_pressure, c, t, dS, eqn)
{
#if !RP_HOST
    Thread *ta, *ts;
    real dens, vof_gx, vof_gy, vof_gz, vof_g, dens_inter, disc, source, vofc, F,
         temp, state;
    real xc[ND_ND];
    real flow_time, torch_loc;

    flow_time = RP_Get_Real("flow-time");
    C_CENTROID(xc, c, t);
    ta = THREAD_SUB_THREAD(t, 0);
    ts = THREAD_SUB_THREAD(t, 1);

    /* Note: z-pressure reads gas-phase VOF from ta (primary phase) */
    vofc       = C_VOF(c, ta);
    dens       = C_R(c, t);
    temp       = C_T(c, t);
    dens_inter = 0.5 * (C_R(c, ta) + C_R(c, ts));

    state = ((temp - 1441) + abs(temp - 1441)) / (2 * abs(temp - 1441) + 1);

    /* Recoil pressure z-component: P_recoil * n_z,  n_z = UDM-2 */
    source = ((1 + 0.08) / 2) * 101325
             * exp(((temp - 2862) / (8.314472 * temp * 2862)) * 796600)
             * C_UDMI(c, t, 2) * dens / dens_inter;

    source = state * source;
    return source;
#endif
}
