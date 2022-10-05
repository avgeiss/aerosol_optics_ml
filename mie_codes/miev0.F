      subroutine  miev0 ( xx, crefin, perfct, mimcut, anyang,
     $                    numang, xmu, nmom, ipolzn, momdim, prnt,
     $                    qext, qsca, gqsc, pmom, sforw, sback, s1,
     $                    s2, tforw, tback )
c
c    computes mie scattering and extinction efficiencies; asymmetry
c    factor;  forward- and backscatter amplitude;  scattering
c    amplitudes for incident polarization parallel and perpendicular
c    to the plane of scattering, as functions of scattering angle;
c    coefficients in the legendre polynomial expansions of either the
c    unpolarized phase function or the polarized phase matrix;
c    and some quantities needed in polarized radiative transfer.
c
c      calls :  biga, ckinmi, small1, small2, testmi, miprnt,
c               lpcoef, errmsg
c
c      i n t e r n a l   v a r i a b l e s
c      -----------------------------------
c
c  an,bn           mie coefficients  little-a-sub-n, little-b-sub-n
c                     ( ref. 1, eq. 16 )
c  anm1,bnm1       mie coefficients  little-a-sub-(n-1),
c                     little-b-sub-(n-1);  used in -gqsc- sum
c  anp             coeffs. in s+ expansion ( ref. 2, p. 1507 )
c  bnp             coeffs. in s- expansion ( ref. 2, p. 1507 )
c  anpm            coeffs. in s+ expansion ( ref. 2, p. 1507 )
c                     when  mu  is replaced by  - mu
c  bnpm            coeffs. in s- expansion ( ref. 2, p. 1507 )
c                     when  mu  is replaced by  - mu
c  calcmo(k)       true, calculate moments for k-th phase quantity
c                     (derived from -ipolzn-; used only in 'lpcoef')
c  cbiga(n)        bessel function ratio capital-a-sub-n (ref. 2, eq. 2)
c                     ( complex version )
c  cior            complex index of refraction with negative
c                     imaginary part (van de hulst convention)
c  cioriv          1 / cior
c  coeff           ( 2n + 1 ) / ( n ( n + 1 ) )
c  fn              floating point version of index in loop performing
c                     mie series summation
c  lita,litb(n)    mie coefficients -an-, -bn-, saved in arrays for
c                     use in calculating legendre moments *pmom*
c  maxtrm          max. possible no. of terms in mie series
c  mm              + 1 and  - 1,  alternately.
c  mim             magnitude of imaginary refractive index
c  mre             real part of refractive index
c  maxang          max. possible value of input variable -numang-
c  nangd2          (numang+1)/2 ( no. of angles in 0-90 deg; anyang=f )
c  noabs           true, sphere non-absorbing (determined by -mimcut-)
c  np1dn           ( n + 1 ) / n
c  npquan          highest-numbered phase quantity for which moments are
c                     to be calculated (the largest digit in -ipolzn-
c                     if  ipolzn .ne. 0)
c  ntrm            no. of terms in mie series
c  pass1           true on first entry, false thereafter; for self-test
c  pin(j)          angular function little-pi-sub-n ( ref. 2, eq. 3 )
c                     at j-th angle
c  pinm1(j)        little-pi-sub-(n-1) ( see -pin- ) at j-th angle
c  psinm1          ricatti-bessel function psi-sub-(n-1), argument -xx-
c  psin            ricatti-bessel function psi-sub-n of argument -xx-
c                     ( ref. 1, p. 11 ff. )
c  rbiga(n)        bessel function ratio capital-a-sub-n (ref. 2, eq. 2)
c                     ( real version, for when imag refrac index = 0 )
c  rioriv          1 / mre
c  rn              1 / n
c  rtmp            (real) temporary variable
c  sp(j)           s+  for j-th angle  ( ref. 2, p. 1507 )
c  sm(j)           s-  for j-th angle  ( ref. 2, p. 1507 )
c  sps(j)          s+  for (numang+1-j)-th angle ( anyang=false )
c  sms(j)          s-  for (numang+1-j)-th angle ( anyang=false )
c  taun            angular function little-tau-sub-n ( ref. 2, eq. 4 )
c                     at j-th angle
c  tcoef           n ( n+1 ) ( 2n+1 ) (for summing tforw,tback series)
c  twonp1          2n + 1
c  yesang          true if scattering amplitudes are to be calculated
c  zetnm1          ricatti-bessel function  zeta-sub-(n-1) of argument
c                     -xx-  ( ref. 2, eq. 17 )
c  zetn            ricatti-bessel function  zeta-sub-n of argument -xx-
c
c ----------------------------------------------------------------------
c --------  i / o specifications for subroutine miev0  -----------------
c ----------------------------------------------------------------------
      implicit none
      logical  anyang, perfct, prnt(*)
      integer  ipolzn, momdim, numang, nmom
      real*8     gqsc, mimcut, pmom( 0:momdim, * ), qext, qsca,
     $         xmu(*), xx
      complex*16  crefin, sforw, sback, s1(*), s2(*), tforw(*),
     $         tback(*)
      integer maxang,mxang2,maxtrm
      real*8 onethr
c ----------------------------------------------------------------------
c
      parameter ( maxang = 501, mxang2 = maxang/2 + 1 )
c
c                                  ** note --  maxtrm = 10100  is neces-
c                                  ** sary to do some of the test probs,
c                                  ** but 1100 is sufficient for most
c                                  ** conceivable applications
      parameter ( maxtrm = 1100 )
      parameter ( onethr = 1./3. )
c
      logical   anysav, calcmo(4), noabs, ok, pass1, persav, yesang
      integer   npquan
      integer i,j,n,nmosav,iposav,numsav,ntrm,nangd2
      real*8      mim, mimsav, mre, mm, np1dn
      real*8 rioriv,xmusav,xxsav,sq,fn,rn,twonp1,tcoef, coeff
      real*8 xinv,psinm1,chinm1,psin,chin,rtmp,taun
      real*8      rbiga( maxtrm ), pin( maxang ), pinm1( maxang )
      complex*16   an, bn, anm1, bnm1, anp, bnp, anpm, bnpm, cresav,
     $          cior, cioriv, ctmp, zet, zetnm1, zetn
      complex*16   cbiga( maxtrm ), lita( maxtrm ), litb( maxtrm ),
     $          sp( maxang ), sm( maxang ), sps( mxang2 ), sms( mxang2 )
      equivalence  ( cbiga, rbiga )
      save  pass1
      sq( ctmp ) = dble( ctmp )**2 + dimag( ctmp )**2
      data  pass1 / .true. /
c
c
      if ( pass1 )  then
c                                   ** save certain user input values
         xxsav  = xx
         cresav = crefin
         mimsav = mimcut
         persav = perfct
         anysav = anyang
         nmosav = nmom
         iposav = ipolzn
         numsav = numang
         xmusav = xmu( 1 )
c                              ** reset input values for test case
         xx      = 10.0
         crefin  = ( 1.5, - 0.1 )
         perfct  = .false.
         mimcut  = 0.0
         anyang  = .true.
         numang  = 1
         xmu( 1 )= - 0.7660444
         nmom    = 1
         ipolzn  = - 1
c
      end if
c                                        ** check input and calculate
c                                        ** certain variables from input
c
   10 call  ckinmi( numang, maxang, xx, perfct, crefin, momdim,
     $              nmom, ipolzn, anyang, xmu, calcmo, npquan )
c
      if ( perfct .and. xx .le. 0.1 )  then
c                                            ** use totally-reflecting
c                                            ** small-particle limit
c
         call  small1 ( xx, numang, xmu, qext, qsca, gqsc, sforw,
     $                  sback, s1, s2, tforw, tback, lita, litb )
         ntrm = 2
         go to 200
      end if
c
      if ( .not.perfct )  then
c
         cior = crefin
         if ( dimag( cior ) .gt. 0.0 )  cior = dconjg( cior )
         mre =     dble( cior )
         mim =  - dimag( cior )
         noabs = mim .le. mimcut
         cioriv = 1.0 / cior
         rioriv = 1.0 / mre
c
         if ( xx * dmax1( 1.d0, cdabs(cior) ) .le. 0.d1 ) then
c
c                                    ** use general-refractive-index
c                                    ** small-particle limit
c                                    ** ( ref. 2, p. 1508 )
c
            call  small2 ( xx, cior, .not.noabs, numang, xmu, qext,
     $                     qsca, gqsc, sforw, sback, s1, s2, tforw,
     $                     tback, lita, litb )
            ntrm = 2
            go to 200
         end if
c
      end if
c
      nangd2 = ( numang + 1 ) / 2
      yesang = numang .gt. 0
c                              ** estimate number of terms in mie series
c                              ** ( ref. 2, p. 1508 )
      if ( xx.le.8.0 )  then
         ntrm = xx + 4. * xx**onethr + 1.
      else if ( xx.lt.4200. )  then
         ntrm = xx + 4.05 * xx**onethr + 2.
      else
         ntrm = xx + 4. * xx**onethr + 2.
      end if
      if ( ntrm+1 .gt. maxtrm )
     $     call errmsg( 'miev0--parameter maxtrm too small', .true. )
c
c                            ** calculate logarithmic derivatives of
c                            ** j-bessel-fcn., big-a-sub-(1 to ntrm)
      if ( .not.perfct )
     $     call  biga( cior, xx, ntrm, noabs, yesang, rbiga, cbiga )
c
c                            ** initialize ricatti-bessel functions
c                            ** (psi,chi,zeta)-sub-(0,1) for upward
c                            ** recurrence ( ref. 1, eq. 19 )
      xinv = 1.0 / xx
      psinm1   = dsin( xx )
      chinm1   = dcos( xx )
      psin = psinm1 * xinv - chinm1
      chin = chinm1 * xinv + psinm1
      zetnm1 = dcmplx( psinm1, chinm1 )
      zetn   = dcmplx( psin, chin )
c                                     ** initialize previous coeffi-
c                                     ** cients for -gqsc- series
      anm1 = ( 0.0, 0.0 )
      bnm1 = ( 0.0, 0.0 )
c                             ** initialize angular function little-pi
c                             ** and sums for s+, s- ( ref. 2, p. 1507 )
      if ( anyang )  then
         do  60  j = 1, numang
            pinm1( j ) = 0.0
            pin( j )   = 1.0
            sp ( j ) = ( 0.0, 0.0 )
            sm ( j ) = ( 0.0, 0.0 )
   60    continue
      else
         do  70  j = 1, nangd2
            pinm1( j ) = 0.0
            pin( j )   = 1.0
            sp ( j ) = ( 0.0, 0.0 )
            sm ( j ) = ( 0.0, 0.0 )
            sps( j ) = ( 0.0, 0.0 )
            sms( j ) = ( 0.0, 0.0 )
   70    continue
      end if
c                         ** initialize mie sums for efficiencies, etc.
      qsca = 0.0
      gqsc = 0.0
      sforw      = ( 0., 0. )
      sback      = ( 0., 0. )
      tforw( 1 ) = ( 0., 0. )
      tback( 1 ) = ( 0., 0. )
c
c
c ---------  loop to sum mie series  -----------------------------------
c
      mm = + 1.0
      do  100  n = 1, ntrm
c                           ** compute various numerical coefficients
         fn     = n
         rn     = 1.0 / fn
         np1dn  = 1.0 + rn
         twonp1 = 2 * n + 1
         coeff  = twonp1 / ( fn * ( n + 1 ) )
         tcoef  = twonp1 * ( fn * ( n + 1 ) )
c
c                              ** calculate mie series coefficients
         if ( perfct )  then
c                                   ** totally-reflecting case
c
            an = ( ( fn*xinv ) * psin - psinm1 ) /
     $           ( ( fn*xinv ) * zetn - zetnm1 )
            bn = psin / zetn
c
         else if ( noabs )  then
c                                      ** no-absorption case
c
            an =  ( ( rioriv*rbiga(n) + ( fn*xinv ) ) * psin - psinm1 )
     $          / ( ( rioriv*rbiga(n) + ( fn*xinv ) ) * zetn - zetnm1 )
            bn =  ( (  mre * rbiga(n) + ( fn*xinv ) ) * psin - psinm1 )
     $          / ( (  mre * rbiga(n) + ( fn*xinv ) ) * zetn - zetnm1 )
         else
c                                       ** absorptive case
c
            an = ( ( cioriv * cbiga(n) + ( fn*xinv ) ) * psin - psinm1 )
     $          /( ( cioriv * cbiga(n) + ( fn*xinv ) ) * zetn - zetnm1 )
            bn = ( (   cior * cbiga(n) + ( fn*xinv ) ) * psin - psinm1 )
     $          /( (   cior * cbiga(n) + ( fn*xinv ) ) * zetn - zetnm1 )
            qsca = qsca + twonp1 * ( sq( an ) + sq( bn ) )
c
         end if
c                       ** save mie coefficients for *pmom* calculation
         lita( n ) = an
         litb( n ) = bn
c                            ** increment mie sums for non-angle-
c                            ** dependent quantities
c
         sforw      = sforw      + twonp1 * ( an + bn )
         tforw( 1 ) = tforw( 1 ) + tcoef  * ( an - bn )
         sback      = sback      + ( mm * twonp1 ) * ( an - bn )
         tback( 1 ) = tback( 1 ) + ( mm * tcoef )  * ( an + bn )
         gqsc = gqsc + ( fn - rn ) * dble( anm1 * dconjg( an )
     $                                   + bnm1 * dconjg( bn ) )
     $          + coeff * dble( an * dconjg( bn ) )
c
         if ( yesang )  then
c                                      ** put mie coefficients in form
c                                      ** needed for computing s+, s-
c                                      ** ( ref. 2, p. 1507 )
            anp = coeff * ( an + bn )
            bnp = coeff * ( an - bn )
c                                      ** increment mie sums for s+, s-
c                                      ** while upward recursing
c                                      ** angular functions little pi
c                                      ** and little tau
            if ( anyang )  then
c                                         ** arbitrary angles
c
c                                              ** vectorizable loop
               do  80  j = 1, numang
                  rtmp = ( xmu( j ) * pin( j ) ) - pinm1( j )
                  taun =  fn * rtmp - pinm1( j )
                  sp( j )  = sp( j ) + anp * ( pin( j ) + taun )
                  sm( j )  = sm( j ) + bnp * ( pin( j ) - taun )
                  pinm1( j ) = pin( j )
                  pin( j ) = ( xmu( j ) * pin( j ) ) + np1dn * rtmp
   80          continue
c
            else
c                                  ** angles symmetric about 90 degrees
               anpm = mm * anp
               bnpm = mm * bnp
c                                          ** vectorizable loop
               do  90  j = 1, nangd2
                  rtmp = ( xmu( j ) * pin( j ) ) - pinm1( j )
                  taun =  fn * rtmp - pinm1( j )
                  sp ( j ) = sp ( j ) +  anp * ( pin( j ) + taun )
                  sms( j ) = sms( j ) + bnpm * ( pin( j ) + taun )
                  sm ( j ) = sm ( j ) +  bnp * ( pin( j ) - taun )
                  sps( j ) = sps( j ) + anpm * ( pin( j ) - taun )
                  pinm1( j ) = pin( j )
                  pin( j ) = ( xmu( j ) * pin( j ) ) + np1dn * rtmp
   90          continue
c
            end if
         end if
c                          ** update relevant quantities for next
c                          ** pass through loop
         mm   =  - mm
         anm1 = an
         bnm1 = bn
c                           ** upward recurrence for ricatti-bessel
c                           ** functions ( ref. 1, eq. 17 )
c
         zet    = ( twonp1 * xinv ) * zetn - zetnm1
         zetnm1 = zetn
         zetn   = zet
         psinm1 = psin
         psin   = dble( zetn )
  100 continue
c
c ---------- end loop to sum mie series --------------------------------
c
c
      qext = 2. / xx**2 * dble( sforw )
      if ( perfct .or. noabs )  then
         qsca = qext
      else
         qsca = 2. / xx**2 * qsca
      end if
c
      gqsc = 4. / xx**2 * gqsc
      sforw = 0.5 * sforw
      sback = 0.5 * sback
      tforw( 2 ) = 0.5 * (   sforw + 0.25 * tforw( 1 ) )
      tforw( 1 ) = 0.5 * (   sforw - 0.25 * tforw( 1 ) )
      tback( 2 ) = 0.5 * (   sback + 0.25 * tback( 1 ) )
      tback( 1 ) = 0.5 * ( - sback + 0.25 * tback( 1 ) )
c
      if ( yesang )  then
c                                ** recover scattering amplitudes
c                                ** from s+, s- ( ref. 1, eq. 11 )
         if ( anyang )  then
c                                         ** vectorizable loop
            do  110  j = 1, numang
               s1( j ) = 0.5 * ( sp( j ) + sm( j ) )
               s2( j ) = 0.5 * ( sp( j ) - sm( j ) )
  110       continue
c
         else
c                                         ** vectorizable loop
            do  120  j = 1, nangd2
               s1( j ) = 0.5 * ( sp( j ) + sm( j ) )
               s2( j ) = 0.5 * ( sp( j ) - sm( j ) )
  120       continue
c                                         ** vectorizable loop
            do  130  j = 1, nangd2
               s1( numang+1 - j ) = 0.5 * ( sps( j ) + sms( j ) )
               s2( numang+1 - j ) = 0.5 * ( sps( j ) - sms( j ) )
  130       continue
         end if
c
      end if
c                                         ** calculate legendre moments
  200 if ( nmom.gt.0 )
     $     call lpcoef ( ntrm, nmom, ipolzn, momdim, calcmo, npquan,
     $                   lita, litb, pmom )
c
      if ( dimag(crefin) .gt. 0.0 )  then
c                                         ** take complex conjugates
c                                         ** of scattering amplitudes
         sforw = dconjg( sforw )
         sback = dconjg( sback )
         do  210  i = 1, 2
            tforw( i ) = dconjg( tforw(i) )
            tback( i ) = dconjg( tback(i) )
  210    continue
c
         do  220  j = 1, numang
            s1( j ) = dconjg( s1(j) )
            s2( j ) = dconjg( s2(j) )
  220    continue
c
      end if
c
      if ( pass1 )  then
c                             ** compare test case results with
c                             ** correct answers and abort if bad
c
         call  testmi ( qext, qsca, gqsc, sforw, sback, s1, s2,
     $                  tforw, tback, pmom, momdim, ok )
         if ( .not. ok )  then
            prnt(1) = .false.
            prnt(2) = .false.
            call  miprnt( prnt, xx, perfct, crefin, numang, xmu, qext,
     $                    qsca, gqsc, nmom, ipolzn, momdim, calcmo,
     $                    pmom, sforw, sback, tforw, tback, s1, s2 )
            call errmsg( 'miev0 -- self-test failed', .true. )
         end if
c                                       ** restore user input values
         xx     = xxsav
         crefin = cresav
         mimcut = mimsav
         perfct = persav
         anyang = anysav
         nmom   = nmosav
         ipolzn = iposav
         numang = numsav
         xmu(1) = xmusav
         pass1 = .false.
         go to 10
c
      end if
c
      if ( prnt(1) .or. prnt(2) )
     $   call  miprnt( prnt, xx, perfct, crefin, numang, xmu, qext,
     $                 qsca, gqsc, nmom, ipolzn, momdim, calcmo,
     $                 pmom, sforw, sback, tforw, tback, s1, s2 )
c
      return
c
      end
      subroutine  ckinmi( numang, maxang, xx, perfct, crefin, momdim,
     $                    nmom, ipolzn, anyang, xmu, calcmo, npquan )
c
c        check for bad input to 'miev0' and calculate -calcmo,npquan-
c
      implicit none
      logical  perfct, anyang, calcmo(*)
      integer  numang, maxang, momdim, nmom, ipolzn, npquan
      real*8    xx, xmu(*)
      integer i,l,j,ip
      complex*16  crefin
c
      character*4  string
      logical  inperr
c
      inperr = .false.
c
      if ( numang.gt.maxang )  then
         call errmsg( 'miev0--parameter maxang too small', .true. )
         inperr = .true.
      end if
      if ( numang.lt.0 )  call  wrtbad( 'numang', inperr )
      if ( xx.lt.0. )     call  wrtbad( 'xx', inperr )
      if ( .not.perfct .and. dble(crefin).le.0. )then
         print *,'crefin=',crefin
         call wrtbad( 'crefin', inperr )
      endif
      if ( momdim.lt.1 )  call wrtbad( 'momdim', inperr )
c
      if ( nmom.ne.0 )  then
         if ( nmom.lt.0 .or. nmom.gt.momdim ) call wrtbad('nmom',inperr)
         if ( iabs(ipolzn).gt.4444 )  call  wrtbad( 'ipolzn', inperr )
         npquan = 0
         do 5  l = 1, 4
            calcmo( l ) = .false.
    5    continue
         if ( ipolzn.ne.0 )  then
c                                 ** parse out -ipolzn- into its digits
c                                 ** to find which phase quantities are
c                                 ** to have their moments calculated
c
            write( string, '(i4)' )  iabs(ipolzn)
            do 10  j = 1, 4
               ip = ichar( string(j:j) ) - ichar( '0' )
               if ( ip.ge.1 .and. ip.le.4 )  calcmo( ip ) = .true.
               if ( ip.eq.0 .or. (ip.ge.5 .and. ip.le.9) )
     $              call  wrtbad( 'ipolzn', inperr )
               npquan = max0( npquan, ip )
   10       continue
         end if
      end if
c
      if ( anyang )  then
c                                ** allow for slight imperfections in
c                                ** computation of cosine
          do  20  i = 1, numang
             if ( xmu(i).lt.-1.00001 .or. xmu(i).gt.1.00001 )
     $            call wrtbad( 'xmu', inperr )
   20     continue
      else
          do  22  i = 1, ( numang + 1 ) / 2
             if ( xmu(i).lt.-0.00001 .or. xmu(i).gt.1.00001 )
     $            call wrtbad( 'xmu', inperr )
   22     continue
      end if
c
      if ( inperr )
     $     call errmsg( 'miev0--input error(s).  aborting...', .true. )
c
      if ( xx.gt.20000.0 .or. dble(crefin).gt.10.0 .or.
     $     dabs( dimag(crefin) ).gt.10.0 )  call  errmsg(
     $     'miev0--xx or crefin outside tested range', .false. )
c
      return
      end
      subroutine  lpcoef ( ntrm, nmom, ipolzn, momdim, calcmo, npquan,
     $                     a, b, pmom )
c
c         calculate legendre polynomial expansion coefficients (also
c         called moments) for phase quantities ( ref. 5 formulation )
c
c     input:  ntrm                    number terms in mie series
c             nmom, ipolzn, momdim    'miev0' arguments
c             calcmo                  flags calculated from -ipolzn-
c             npquan                  defined in 'miev0'
c             a, b                    mie series coefficients
c
c     output: pmom                   legendre moments ('miev0' argument)
c
c     *** notes ***
c
c         (1)  eqs. 2-5 are in error in dave, appl. opt. 9,
c         1888 (1970).  eq. 2 refers to m1, not m2;  eq. 3 refers to
c         m2, not m1.  in eqs. 4 and 5, the subscripts on the second
c         term in square brackets should be interchanged.
c
c         (2)  the general-case logic in this subroutine works correctly
c         in the two-term mie series case, but subroutine  'lpco2t'
c         is called instead, for speed.
c
c         (3)  subroutine  'lpco1t', to do the one-term case, is never
c         called within the context of 'miev0', but is included for
c         complete generality.
c
c         (4)  some improvement in speed is obtainable by combining the
c         310- and 410-loops, if moments for both the third and fourth
c         phase quantities are desired, because the third phase quantity
c         is the real part of a complex series, while the fourth phase
c         quantity is the imaginary part of that very same series.  but
c         most users are not interested in the fourth phase quantity,
c         which is related to circular polarization, so the present
c         scheme is usually more efficient.
c
      implicit none
      logical  calcmo(*)
      integer  ipolzn, momdim, nmom, ntrm, npquan
      real*8    pmom( 0:momdim, * )
      complex*16  a(*), b(*)
c
c           ** specification of local variables
c
c      am(m)       numerical coefficients  a-sub-m-super-l
c                     in dave, eqs. 1-15, as simplified in ref. 5.
c
c      bi(i)       numerical coefficients  b-sub-i-super-l
c                     in dave, eqs. 1-15, as simplified in ref. 5.
c
c      bidel(i)    1/2 bi(i) times factor capital-del in dave
c
c      cm,dm()     arrays c and d in dave, eqs. 16-17 (mueller form),
c                     calculated using recurrence derived in ref. 5
c
c      cs,ds()     arrays c and d in ref. 4, eqs. a5-a6 (sekera form),
c                     calculated using recurrence derived in ref. 5
c
c      c,d()       either -cm,dm- or -cs,ds-, depending on -ipolzn-
c
c      evenl       true for even-numbered moments;  false otherwise
c
c      idel        1 + little-del  in dave
c
c      maxtrm      max. no. of terms in mie series
c
c      maxmom      max. no. of non-zero moments
c
c      nummom      number of non-zero moments
c
c      recip(k)    1 / k
c
      integer maxtrm,maxmom,mxmom2,maxrcp
      parameter  ( maxtrm = 1102, maxmom = 2*maxtrm, mxmom2 = maxmom/2,
     $             maxrcp = 4*maxtrm + 2 )
      real*8      am( 0:maxtrm ), bi( 0:mxmom2 ), bidel( 0:mxmom2 ),
     $           recip( maxrcp )
      complex*16 cm( maxtrm ), dm( maxtrm ), cs( maxtrm ), ds( maxtrm ),
     $           c( maxtrm ), d( maxtrm )
      integer k,j,l,nummom,ld2,idel,m,i,mmax,imax
      real*8 sum
      equivalence  ( c, cm ),  ( d, dm )
      logical    pass1, evenl
      save  pass1, recip
      data  pass1 / .true. /
c
c
      if ( pass1 )  then
c
         do  1  k = 1, maxrcp
            recip( k ) = 1.0 / k
    1    continue
         pass1 = .false.
c
      end if
c
      do  5  j = 1, max0( 1, npquan )
         do  5  l = 0, nmom
            pmom( l, j ) = 0.0
    5 continue
c
      if ( ntrm.eq.1 )  then
         call  lpco1t ( nmom, ipolzn, momdim, calcmo, a, b, pmom )
         return
      else if ( ntrm.eq.2 )  then
         call  lpco2t ( nmom, ipolzn, momdim, calcmo, a, b, pmom )
         return
      end if
c
      if ( ntrm+2 .gt. maxtrm )
     $     call errmsg( 'lpcoef--parameter maxtrm too small', .true. )
c
c                                     ** calculate mueller c, d arrays
      cm( ntrm+2 ) = ( 0., 0. )
      dm( ntrm+2 ) = ( 0., 0. )
      cm( ntrm+1 ) = ( 1. - recip( ntrm+1 ) ) * b( ntrm )
      dm( ntrm+1 ) = ( 1. - recip( ntrm+1 ) ) * a( ntrm )
      cm( ntrm ) = ( recip(ntrm) + recip(ntrm+1) ) * a( ntrm )
     $             + ( 1. - recip(ntrm) ) * b( ntrm-1 )
      dm( ntrm ) = ( recip(ntrm) + recip(ntrm+1) ) * b( ntrm )
     $             + ( 1. - recip(ntrm) ) * a( ntrm-1 )
c
      do  10  k = ntrm-1, 2, -1
         cm( k ) = cm( k+2 ) - ( 1. + recip(k+1) ) * b( k+1 )
     $                       + ( recip(k) + recip(k+1) ) * a( k )
     $                       + ( 1. - recip(k) ) * b( k-1 )
         dm( k ) = dm( k+2 ) - ( 1. + recip(k+1) ) * a( k+1 )
     $                       + ( recip(k) + recip(k+1) ) * b( k )
     $                       + ( 1. - recip(k) ) * a( k-1 )
   10 continue
      cm( 1 ) = cm( 3 ) + 1.5 * ( a( 1 ) - b( 2 ) )
      dm( 1 ) = dm( 3 ) + 1.5 * ( b( 1 ) - a( 2 ) )
c
      if ( ipolzn.ge.0 )  then
c
         do  20  k = 1, ntrm + 2
            c( k ) = ( 2*k - 1 ) * cm( k )
            d( k ) = ( 2*k - 1 ) * dm( k )
   20    continue
c
      else
c                                    ** compute sekera c and d arrays
         cs( ntrm+2 ) = ( 0., 0. )
         ds( ntrm+2 ) = ( 0., 0. )
         cs( ntrm+1 ) = ( 0., 0. )
         ds( ntrm+1 ) = ( 0., 0. )
c
         do  30  k = ntrm, 1, -1
            cs( k ) = cs( k+2 ) + ( 2*k + 1 ) * ( cm( k+1 ) - b( k ) )
            ds( k ) = ds( k+2 ) + ( 2*k + 1 ) * ( dm( k+1 ) - a( k ) )
   30    continue
c
         do  40  k = 1, ntrm + 2
            c( k ) = ( 2*k - 1 ) * cs( k )
            d( k ) = ( 2*k - 1 ) * ds( k )
   40    continue
c
      end if
c
c
      if( ipolzn.lt.0 )  nummom = min0( nmom, 2*ntrm - 2 )
      if( ipolzn.ge.0 )  nummom = min0( nmom, 2*ntrm )
      if ( nummom .gt. maxmom )
     $     call errmsg( 'lpcoef--parameter maxtrm too small', .true. )
c
c                               ** loop over moments
      do  500  l = 0, nummom
         ld2 = l / 2
         evenl = mod( l,2 ) .eq. 0
c                                    ** calculate numerical coefficients
c                                    ** a-sub-m and b-sub-i in dave
c                                    ** double-sums for moments
         if( l.eq.0 )  then
c
            idel = 1
            do  60  m = 0, ntrm
               am( m ) = 2.0 * recip( 2*m + 1 )
   60       continue
            bi( 0 ) = 1.0
c
         else if( evenl )  then
c
            idel = 1
            do  70  m = ld2, ntrm
               am( m ) = ( 1. + recip( 2*m-l+1 ) ) * am( m )
   70       continue
            do  75  i = 0, ld2-1
               bi( i ) = ( 1. - recip( l-2*i ) ) * bi( i )
   75       continue
            bi( ld2 ) = ( 2. - recip( l ) ) * bi( ld2-1 )
c
         else
c
            idel = 2
            do  80  m = ld2, ntrm
               am( m ) = ( 1. - recip( 2*m+l+2 ) ) * am( m )
   80       continue
            do  85  i = 0, ld2
               bi( i ) = ( 1. - recip( l+2*i+1 ) ) * bi( i )
   85       continue
c
         end if
c                                     ** establish upper limits for sums
c                                     ** and incorporate factor capital-
c                                     ** del into b-sub-i
         mmax = ntrm - idel
         if( ipolzn.ge.0 )  mmax = mmax + 1
         imax = min0( ld2, mmax - ld2 )
         if( imax.lt.0 )  go to 600
         do  90  i = 0, imax
            bidel( i ) = bi( i )
   90    continue
         if( evenl )  bidel( 0 ) = 0.5 * bidel( 0 )
c
c                                    ** perform double sums just for
c                                    ** phase quantities desired by user
         if( ipolzn.eq.0 )  then
c
            do  110  i = 0, imax
c                                           ** vectorizable loop (cray)
               sum = 0.0
               do  100  m = ld2, mmax - i
                  sum = sum + am( m ) *
     $                      ( dble( c(m-i+1) * dconjg( c(m+i+idel) ) )
     $                      + dble( d(m-i+1) * dconjg( d(m+i+idel) ) ) )
  100          continue
               pmom( l,1 ) = pmom( l,1 ) + bidel( i ) * sum
  110       continue
            pmom( l,1 ) = 0.5 * pmom( l,1 )
            go to 500
c
         end if
c
         if ( calcmo(1) )  then
            do  160  i = 0, imax
c                                           ** vectorizable loop (cray)
               sum = 0.0
               do  150  m = ld2, mmax - i
                  sum = sum + am( m ) *
     $                        dble( c(m-i+1) * dconjg( c(m+i+idel) ) )
  150          continue
               pmom( l,1 ) = pmom( l,1 ) + bidel( i ) * sum
  160       continue
         end if
c
c
         if ( calcmo(2) )  then
            do  210  i = 0, imax
c                                           ** vectorizable loop (cray)
               sum = 0.0
               do  200  m = ld2, mmax - i
                  sum = sum + am( m ) *
     $                        dble( d(m-i+1) * dconjg( d(m+i+idel) ) )
  200          continue
               pmom( l,2 ) = pmom( l,2 ) + bidel( i ) * sum
  210       continue
         end if
c
c
         if ( calcmo(3) )  then
            do  310  i = 0, imax
c                                           ** vectorizable loop (cray)
               sum = 0.0
               do  300  m = ld2, mmax - i
                  sum = sum + am( m ) *
     $                      ( dble( c(m-i+1) * dconjg( d(m+i+idel) ) )
     $                      + dble( c(m+i+idel) * dconjg( d(m-i+1) ) ) )
  300          continue
               pmom( l,3 ) = pmom( l,3 ) + bidel( i ) * sum
  310       continue
            pmom( l,3 ) = 0.5 * pmom( l,3 )
         end if
c
c
         if ( calcmo(4) )  then
            do  410  i = 0, imax
c                                           ** vectorizable loop (cray)
               sum = 0.0
               do  400  m = ld2, mmax - i
                  sum = sum + am( m ) *
     $                      ( dimag( c(m-i+1) * dconjg( d(m+i+idel) ) )
     $                      + dimag( c(m+i+idel) * dconjg( d(m-i+1) ) ))
  400          continue
               pmom( l,4 ) = pmom( l,4 ) + bidel( i ) * sum
  410       continue
            pmom( l,4 ) = - 0.5 * pmom( l,4 )
         end if
c
  500 continue
c
c
  600 return
      end
      subroutine  lpco1t ( nmom, ipolzn, momdim, calcmo, a, b, pmom )
c
c         calculate legendre polynomial expansion coefficients (also
c         called moments) for phase quantities in special case where
c         no. terms in mie series = 1
c
c        input:  nmom, ipolzn, momdim     'miev0' arguments
c                calcmo                   flags calculated from -ipolzn-
c                a(1), b(1)               mie series coefficients
c
c        output: pmom                     legendre moments
c
      implicit none
      logical  calcmo(*)
      integer  ipolzn, momdim, nmom,nummom,l
      real*8    pmom( 0:momdim, * ),sq,a1sq,b1sq
      complex*16  a(*), b(*), ctmp, a1b1c
      sq( ctmp ) = dble( ctmp )**2 + dimag( ctmp )**2
c
c
      a1sq = sq( a(1) )
      b1sq = sq( b(1) )
      a1b1c = a(1) * dconjg( b(1) )
c
      if( ipolzn.lt.0 )  then
c
         if( calcmo(1) )  pmom( 0,1 ) = 2.25 * b1sq
         if( calcmo(2) )  pmom( 0,2 ) = 2.25 * a1sq
         if( calcmo(3) )  pmom( 0,3 ) = 2.25 * dble( a1b1c )
         if( calcmo(4) )  pmom( 0,4 ) = 2.25 *dimag( a1b1c )
c
      else
c
         nummom = min0( nmom, 2 )
c                                   ** loop over moments
         do  100  l = 0, nummom
c
            if( ipolzn.eq.0 )  then
               if( l.eq.0 )  pmom( l,1 ) = 1.5 * ( a1sq + b1sq )
               if( l.eq.1 )  pmom( l,1 ) = 1.5 * dble( a1b1c )
               if( l.eq.2 )  pmom( l,1 ) = 0.15 * ( a1sq + b1sq )
               go to 100
            end if
c
            if( calcmo(1) )  then
               if( l.eq.0 )  pmom( l,1 ) = 2.25 * ( a1sq + b1sq / 3. )
               if( l.eq.1 )  pmom( l,1 ) = 1.5 * dble( a1b1c )
               if( l.eq.2 )  pmom( l,1 ) = 0.3 * b1sq
            end if
c
            if( calcmo(2) )  then
               if( l.eq.0 )  pmom( l,2 ) = 2.25 * ( b1sq + a1sq / 3. )
               if( l.eq.1 )  pmom( l,2 ) = 1.5 * dble( a1b1c )
               if( l.eq.2 )  pmom( l,2 ) = 0.3 * a1sq
            end if
c
            if( calcmo(3) )  then
               if( l.eq.0 )  pmom( l,3 ) = 3.0 * dble( a1b1c )
               if( l.eq.1 )  pmom( l,3 ) = 0.75 * ( a1sq + b1sq )
               if( l.eq.2 )  pmom( l,3 ) = 0.3 * dble( a1b1c )
            end if
c
            if( calcmo(4) )  then
               if( l.eq.0 )  pmom( l,4 ) = - 1.5 * dimag( a1b1c )
               if( l.eq.1 )  pmom( l,4 ) = 0.0
               if( l.eq.2 )  pmom( l,4 ) = 0.3 * dimag( a1b1c )
            end if
c
  100    continue
c
      end if
c
      return
      end
      subroutine  lpco2t ( nmom, ipolzn, momdim, calcmo, a, b, pmom )
c
c         calculate legendre polynomial expansion coefficients (also
c         called moments) for phase quantities in special case where
c         no. terms in mie series = 2
c
c        input:  nmom, ipolzn, momdim     'miev0' arguments
c                calcmo                   flags calculated from -ipolzn-
c                a(1-2), b(1-2)           mie series coefficients
c
c        output: pmom                     legendre moments
c
      implicit none
      logical  calcmo(*)
      integer  ipolzn, momdim, nmom,l,nummom
      real*8    pmom( 0:momdim, * ),sq,a1sq,b1sq,pm1,pm2,a2sq,b2sq
      complex*16  a(*), b(*)
      complex*16  a2c, b2c, ctmp, ca, cac, cat, cb, cbc, cbt, cg, ch
      sq( ctmp ) = dble( ctmp )**2 + dimag( ctmp )**2
c
c
      ca = 3. * a(1) - 5. * b(2)
      cat= 3. * b(1) - 5. * a(2)
      cac = dconjg( ca )
      a2sq = sq( a(2) )
      b2sq = sq( b(2) )
      a2c = dconjg( a(2) )
      b2c = dconjg( b(2) )
c
      if( ipolzn.lt.0 )  then
c                                   ** loop over sekera moments
         nummom = min0( nmom, 2 )
         do  50  l = 0, nummom
c
            if( calcmo(1) )  then
               if( l.eq.0 ) pmom( l,1 ) = 0.25 * ( sq(cat) +
     $                                             (100./3.) * b2sq )
               if( l.eq.1 ) pmom( l,1 ) = (5./3.) * dble( cat * b2c )
               if( l.eq.2 ) pmom( l,1 ) = (10./3.) * b2sq
            end if
c
            if( calcmo(2) )  then
               if( l.eq.0 ) pmom( l,2 ) = 0.25 * ( sq(ca) +
     $                                             (100./3.) * a2sq )
               if( l.eq.1 ) pmom( l,2 ) = (5./3.) * dble( ca * a2c )
               if( l.eq.2 ) pmom( l,2 ) = (10./3.) * a2sq
            end if
c
            if( calcmo(3) )  then
               if( l.eq.0 ) pmom( l,3 ) = 0.25 * dble( cat*cac +
     $                                           (100./3.)*b(2)*a2c )
               if( l.eq.1 ) pmom( l,3 ) = 5./6. * dble( b(2)*cac +
     $                                                  cat*a2c )
               if( l.eq.2 ) pmom( l,3 ) = 10./3. * dble( b(2) * a2c )
            end if
c
            if( calcmo(4) )  then
               if( l.eq.0 ) pmom( l,4 ) = -0.25 * dimag( cat*cac +
     $                                           (100./3.)*b(2)*a2c )
               if( l.eq.1 ) pmom( l,4 ) = -5./6. * dimag( b(2)*cac +
     $                                                  cat*a2c )
               if( l.eq.2 ) pmom( l,4 ) = -10./3. * dimag( b(2) * a2c )
            end if
c
   50    continue
c
      else
c
         cb = 3. * b(1) + 5. * a(2)
         cbt= 3. * a(1) + 5. * b(2)
         cbc = dconjg( cb )
         cg = ( cbc*cbt + 10.*( cac*a(2) + b2c*cat) ) / 3.
         ch = 2.*( cbc*a(2) + b2c*cbt )
c
c                                   ** loop over mueller moments
         nummom = min0( nmom, 4 )
         do  100  l = 0, nummom
c
            if( ipolzn.eq.0 .or. calcmo(1) )  then
               if( l.eq.0 ) pm1 = 0.25 * sq(ca) + sq(cb) / 12.
     $                            + (5./3.) * dble(ca*b2c) + 5.*b2sq
               if( l.eq.1 ) pm1 = dble( cb * ( cac/6. + b2c ) )
               if( l.eq.2 ) pm1 = sq(cb)/30. + (20./7.) * b2sq
     $                            + (2./3.) * dble( ca * b2c )
               if( l.eq.3 ) pm1 = (2./7.) * dble( cb * b2c )
               if( l.eq.4 ) pm1 = (40./63.) * b2sq
               if ( calcmo(1) )  pmom( l,1 ) = pm1
            end if
c
            if( ipolzn.eq.0 .or. calcmo(2) )  then
               if( l.eq.0 ) pm2 = 0.25*sq(cat) + sq(cbt) / 12.
     $                            + (5./3.) * dble(cat*a2c) + 5.*a2sq
               if( l.eq.1 ) pm2 = dble( cbt * ( dconjg(cat)/6. + a2c) )
               if( l.eq.2 ) pm2 = sq(cbt)/30. + (20./7.) * a2sq
     $                            + (2./3.) * dble( cat * a2c )
               if( l.eq.3 ) pm2 = (2./7.) * dble( cbt * a2c )
               if( l.eq.4 ) pm2 = (40./63.) * a2sq
               if ( calcmo(2) )  pmom( l,2 ) = pm2
            end if
c
            if( ipolzn.eq.0 )  then
               pmom( l,1 ) = 0.5 * ( pm1 + pm2 )
               go to 100
            end if
c
            if( calcmo(3) )  then
               if( l.eq.0 ) pmom( l,3 ) = 0.25 * dble( cac*cat + cg +
     $                                                 20.*b2c*a(2) )
               if( l.eq.1 ) pmom( l,3 ) = dble( cac*cbt + cbc*cat +
     $                                          3.*ch ) / 12.
               if( l.eq.2 ) pmom( l,3 ) = 0.1 * dble( cg + (200./7.) *
     $                                                b2c * a(2) )
               if( l.eq.3 ) pmom( l,3 ) = dble( ch ) / 14.
               if( l.eq.4 ) pmom( l,3 ) = 40./63. * dble( b2c * a(2) )
            end if
c
            if( calcmo(4) )  then
               if( l.eq.0 ) pmom( l,4 ) = 0.25 * dimag( cac*cat + cg +
     $                                                  20.*b2c*a(2) )
               if( l.eq.1 ) pmom( l,4 ) = dimag( cac*cbt + cbc*cat +
     $                                           3.*ch ) / 12.
               if( l.eq.2 ) pmom( l,4 ) = 0.1 * dimag( cg + (200./7.) *
     $                                                 b2c * a(2) )
               if( l.eq.3 ) pmom( l,4 ) = dimag( ch ) / 14.
               if( l.eq.4 ) pmom( l,4 ) = 40./63. * dimag( b2c * a(2) )
            end if
c
  100    continue
c
      end if
c
      return
      end
      subroutine  biga( cior, xx, ntrm, noabs, yesang, rbiga, cbiga )
c
c        calculate logarithmic derivatives of j-bessel-function
c
c     input :  cior, xx, ntrm, noabs, yesang  (defined in 'miev0')
c
c    output :  rbiga or cbiga  (defined in 'miev0')
c
c    internal variables :
c
c       confra     value of lentz continued fraction for -cbiga(ntrm)-,
c                     used to initialize downward recurrence.
c       down       = true, use down-recurrence.  false, do not.
c       f1,f2,f3   arithmetic statement functions used in determining
c                     whether to use up-  or down-recurrence
c                     ( ref. 2, eqs. 6-8 )
c       mre        real refractive index
c       mim        imaginary refractive index
c       rezinv     1 / ( mre * xx ); temporary variable for recurrence
c       zinv       1 / ( cior * xx ); temporary variable for recurrence
c
      implicit none
      logical  down, noabs, yesang
      integer  ntrm,n
      real*8    mre, mim, rbiga(*), xx, rezinv, rtmp, f1,f2,f3
      complex*16  cior, ctmp, confra, cbiga(*), zinv
      f1( mre ) =  - 8.0 + mre**2 * ( 26.22 + mre * ( - 0.4474
     $             + mre**3 * ( 0.00204 - 0.000175 * mre ) ) )
      f2( mre ) = 3.9 + mre * ( - 10.8 + 13.78 * mre )
      f3( mre ) =  - 15.04 + mre * ( 8.42 + 16.35 * mre )
c
c                                  ** decide whether 'biga' can be
c                                  ** calculated by up-recurrence
      mre =  dble( cior )
      mim =  dabs( dimag( cior ) )
      if ( mre.lt.1.0 .or. mre.gt.10.0 .or. mim.gt.10.0 )  then
         down = .true.
      else if ( yesang )  then
         down = .true.
         if ( mim*xx .lt. f2( mre ) )  down = .false.
      else
         down = .true.
         if ( mim*xx .lt. f1( mre ) )  down = .false.
      end if
c
      zinv  = 1.0 / ( cior * xx )
      rezinv = 1.0 / ( mre * xx )
c
      if ( down )  then
c                          ** compute initial high-order 'biga' using
c                          ** lentz method ( ref. 1, pp. 17-20 )
c
         ctmp = confra( ntrm, zinv, xx )
c
c                                   *** downward recurrence for 'biga'
c                                   *** ( ref. 1, eq. 22 )
         if ( noabs )  then
c                                            ** no-absorption case
            rbiga( ntrm ) = dble( ctmp )
            do  25  n = ntrm, 2, - 1
               rbiga( n-1 ) = (n*rezinv)
     $                         - 1.0 / ( (n*rezinv) + rbiga( n ) )
   25       continue
c
         else
c                                            ** absorptive case
            cbiga( ntrm ) = ctmp
            do  30  n = ntrm, 2, - 1
               cbiga( n-1 ) = (n*zinv) - 1.0 / ( (n*zinv) + cbiga( n ) )
   30       continue
c
         end if
c
      else
c                              *** upward recurrence for 'biga'
c                              *** ( ref. 1, eqs. 20-21 )
         if ( noabs )  then
c                                            ** no-absorption case
            rtmp = dsin( mre*xx )
            rbiga( 1 ) =  - rezinv
     $                     + rtmp / ( rtmp*rezinv - dcos( mre*xx ) )
            do  40  n = 2, ntrm
               rbiga( n ) = - ( n*rezinv )
     $                       + 1.0 / ( ( n*rezinv ) - rbiga( n-1 ) )
   40       continue
c
         else
c                                                ** absorptive case
            ctmp = cdexp( - dcmplx(0.d0,2.d0) * cior * xx )
            cbiga( 1 ) = - zinv + (1.-ctmp) / ( zinv * (1.-ctmp) -
     $                     dcmplx(0.d0,1.d0)*(1.+ctmp) )
            do  50  n = 2, ntrm
               cbiga( n ) = - (n*zinv) + 1.0 / ((n*zinv) - cbiga( n-1 ))
   50       continue
         end if
c
      end if
c
      return
      end
      complex*16 function  confra( n, zinv, xx )
c
c         compute bessel function ratio capital-a-sub-n from its
c         continued fraction using lentz method ( ref. 1, pp. 17-20 )
c
c         zinv = reciprocal of argument of capital-a
c
c    i n t e r n a l    v a r i a b l e s
c    ------------------------------------
c
c    cak      term in continued fraction expansion of capital-a
c                ( ref. 1, eq. 25 )
c    capt     factor used in lentz iteration for capital-a
c                ( ref. 1, eq. 27 )
c    cdenom   denominator in -capt-  ( ref. 1, eq. 28b )
c    cnumer   numerator   in -capt-  ( ref. 1, eq. 28a )
c    cdtd     product of two successive denominators of -capt-
c                factors  ( ref. 1, eq. 34c )
c    cntn     product of two successive numerators of -capt-
c                factors  ( ref. 1, eq. 34b )
c    eps1     ill-conditioning criterion
c    eps2     convergence criterion
c    kk       subscript k of -cak-  ( ref. 1, eq. 25b )
c    kount    iteration counter ( used only to prevent runaway )
c    maxit    max. allowed no. of iterations
c    mm       + 1  and - 1, alternately
c
      implicit none
      integer   n,maxit,mm,kk,kount
      real*8     xx,eps1,eps2
      complex*16   zinv
      complex*16   cak, capt, cdenom, cdtd, cnumer, cntn
      data  eps1 / 1.e - 2 /, eps2 / 1.e - 8 /
      data  maxit / 10000 /
c
c                                      *** ref. 1, eqs. 25a, 27
      confra = ( n + 1 ) * zinv
      mm     =  - 1
      kk     = 2 * n + 3
      cak    = ( mm * kk ) * zinv
      cdenom = cak
      cnumer = cdenom + 1.0 / confra
      kount  = 1
c
   20 kount = kount + 1
      if ( kount.gt.maxit )
     $     call errmsg( 'confra--iteration failed to converge$', .true.)
c
c                                         *** ref. 2, eq. 25b
      mm  =  - mm
      kk  = kk + 2
      cak = ( mm * kk ) * zinv
c                                         *** ref. 2, eq. 32
      if (      cdabs( cnumer/cak ).le.eps1
     $     .or. cdabs( cdenom/cak ).le.eps1 )  then
c
c                                  ** ill-conditioned case -- stride
c                                  ** two terms instead of one
c
c                                         *** ref. 2, eqs. 34
         cntn   = cak * cnumer + 1.0
         cdtd   = cak * cdenom + 1.0
         confra = ( cntn / cdtd ) * confra
c                                             *** ref. 2, eq. 25b
         mm  =  - mm
         kk  = kk + 2
         cak = ( mm * kk ) * zinv
c                                         *** ref. 2, eqs. 35
         cnumer = cak + cnumer / cntn
         cdenom = cak + cdenom / cdtd
         kount  = kount + 1
         go to 20
c
      else
c                                ** well-conditioned case
c
c                                        *** ref. 2, eqs. 26, 27
         capt   = cnumer / cdenom
         confra = capt * confra
c                                    ** check for convergence
c                                    ** ( ref. 2, eq. 31 )
c
         if (      dabs( dble(capt) - 1.0 ).ge.eps2
     $        .or. dabs( dimag(capt) )      .ge.eps2 )  then
c
c                                        *** ref. 2, eqs. 30a-b
            cnumer = cak + 1.0 / cnumer
            cdenom = cak + 1.0 / cdenom
            go to 20
         end if
      end if
c
      return
c
      end
      subroutine  miprnt( prnt, xx, perfct, crefin, numang, xmu,
     $                    qext, qsca, gqsc, nmom, ipolzn, momdim,
     $                    calcmo, pmom, sforw, sback, tforw, tback,
     $                    s1, s2 )
c
c         print scattering quantities of a single particle
c
      implicit none
      logical  perfct, prnt(*), calcmo(*)
      integer  ipolzn, momdim, nmom, numang,i,m,j
      real*8    gqsc, pmom( 0:momdim, * ), qext, qsca, xx, xmu(*)
      real*8 fi1,fi2,fnorm
      complex*16  crefin, sforw, sback, tforw(*), tback(*), s1(*), s2(*)
      character*22  fmt
c
c
      if ( perfct )  write ( 59, '(''1'',10x,a,1p,e11.4)' )
     $                'perfectly conducting case, size parameter =', xx
      if ( .not.perfct )  write ( 59, '(''1'',10x,3(a,1p,e11.4))' )
     $                  'refractive index:  real ', dble(crefin),
     $             '  imag ', dimag(crefin), ',   size parameter =', xx
c
      if ( prnt(1) .and. numang.gt.0 )  then
c
         write ( 59, '(/,a)' )
     $    '    cos(angle)  ------- s1 ---------  ------- s2 ---------'//
     $    '  --- s1*conjg(s2) ---   i1=s1**2   i2=s2**2  (i1+i2)/2'//
     $    '  deg polzn'
         do  10  i = 1, numang
            fi1 = dble( s1(i) ) **2 + dimag( s1(i) )**2
            fi2 = dble( s2(i) ) **2 + dimag( s2(i) )**2
            write( 59, '( i4, f10.6, 1p,10e11.3 )'   )
     $              i, xmu(i), s1(i), s2(i), s1(i)*dconjg(s2(i)),
     $              fi1, fi2, 0.5*(fi1+fi2), (fi2-fi1)/(fi2+fi1)
   10    continue
c
      end if
c
c
      if ( prnt(2) )  then
c
         write ( 59, '(/,a,9x,a,17x,a,17x,a,/,(0p,f7.2, 1p,6e12.3) )' )
     $           '  angle', 's-sub-1', 't-sub-1', 't-sub-2',
     $               0.0,     sforw,    tforw(1),  tforw(2),
     $              180.,     sback,    tback(1),  tback(2)
         write ( 59, '(/,4(a,1p,e11.4))' )
     $           ' efficiency factors,  extinction:', qext,
     $                              '   scattering:', qsca,
     $                              '   absorption:', qext-qsca,
     $                           '   rad. pressure:', qext-gqsc
c
         if ( nmom.gt.0 )  then
c
            write( 59, '(/,a)' )  ' normalized moments of :'
            if ( ipolzn.eq.0 ) write ( 59, '(''+'',27x,a)' ) 'phase fcn'
            if ( ipolzn.gt.0 )  write ( 59, '(''+'',33x,a)' )
     $         'm1           m2          s21          d21'
            if ( ipolzn.lt.0 )  write ( 59, '(''+'',33x,a)' )
     $         'r1           r2           r3           r4'
c
            fnorm = 4. / ( xx**2 * qsca )
            do  20  m = 0, nmom
               write ( 59, '(a,i4)' )  '      moment no.', m
               do 20  j = 1, 4
                  if( calcmo(j) )  then
                     write( fmt, 98 )  24 + (j-1)*13
                     write ( 59,fmt )  fnorm * pmom(m,j)
                  end if
   20       continue
         end if
c
      end if
c
      return
c
   98 format( '( ''+'', t', i2, ', 1p,e13.4 )' )
      end
      subroutine  small1 ( xx, numang, xmu, qext, qsca, gqsc, sforw,
     $                     sback, s1, s2, tforw, tback, a, b )
c
c       small-particle limit of mie quantities in totally reflecting
c       limit ( mie series truncated after 2 terms )
c
c        a,b       first two mie coefficients, with numerator and
c                  denominator expanded in powers of -xx- ( a factor
c                  of xx**3 is missing but is restored before return
c                  to calling program )  ( ref. 2, p. 1508 )
c
      implicit none
      integer  numang,j
      real*8    gqsc, qext, qsca, xx, xmu(*)
      real*8 twothr,fivthr,fivnin,sq,rtmp
      complex*16  a( 2 ), b( 2 ), sforw, sback, s1(*), s2(*),
     $         tforw(*), tback(*)
c
      parameter  ( twothr = 2./3., fivthr = 5./3., fivnin = 5./9. )
      complex*16    ctmp
      sq( ctmp ) = dble( ctmp )**2 + dimag( ctmp )**2
c
c
      a( 1 ) = dcmplx ( 0.d0, twothr * ( 1. - 0.2 * xx**2 ) )
     $       / dcmplx ( 1.d0 - 0.5 * xx**2, twothr * xx**3 )
c
      b( 1 ) = dcmplx ( 0.d0, - ( 1. - 0.1 * xx**2 ) / 3. )
     $       / dcmplx ( 1.d0 + 0.5 * xx**2, - xx**3 / 3. )
c
      a( 2 ) = dcmplx ( 0.d0,   xx**2 / 30. )
      b( 2 ) = dcmplx ( 0.d0, - xx**2 / 45. )
c
      qsca = 6. * xx**4 * ( sq( a(1) ) + sq( b(1) )
     $                      + fivthr * ( sq( a(2) ) + sq( b(2) ) ) )
      qext = qsca
      gqsc = 6. * xx**4 * dble( a(1) * dconjg( a(2) + b(1) )
     $                    + ( b(1) + fivnin * a(2) ) * dconjg( b(2) ) )
c
      rtmp = 1.5 * xx**3
      sforw      = rtmp * ( a(1) + b(1) + fivthr * ( a(2) + b(2) ) )
      sback      = rtmp * ( a(1) - b(1) - fivthr * ( a(2) - b(2) ) )
      tforw( 1 ) = rtmp * ( b(1) + fivthr * ( 2.*b(2) - a(2) ) )
      tforw( 2 ) = rtmp * ( a(1) + fivthr * ( 2.*a(2) - b(2) ) )
      tback( 1 ) = rtmp * ( b(1) - fivthr * ( 2.*b(2) + a(2) ) )
      tback( 2 ) = rtmp * ( a(1) - fivthr * ( 2.*a(2) + b(2) ) )
c
      do  10  j = 1, numang
         s1( j ) = rtmp * ( a(1) + b(1) * xmu(j) + fivthr *
     $              ( a(2) * xmu(j) + b(2) * ( 2.*xmu(j)**2 - 1. )) )
         s2( j ) = rtmp * ( b(1) + a(1) * xmu(j) + fivthr *
     $              ( b(2) * xmu(j) + a(2) * ( 2.*xmu(j)**2 - 1. )) )
   10 continue
c                                     ** recover actual mie coefficients
      a( 1 ) = xx**3 * a( 1 )
      a( 2 ) = xx**3 * a( 2 )
      b( 1 ) = xx**3 * b( 1 )
      b( 2 ) = xx**3 * b( 2 )
c
      return
      end
      subroutine  small2 ( xx, cior, calcqe, numang, xmu, qext, qsca,
     $                     gqsc, sforw, sback, s1, s2, tforw, tback,
     $                     a, b )
c
c       small-particle limit of mie quantities for general refractive
c       index ( mie series truncated after 2 terms )
c
c        a,b       first two mie coefficients, with numerator and
c                  denominator expanded in powers of -xx- ( a factor
c                  of xx**3 is missing but is restored before return
c                  to calling program )  ( ref. 2, p. 1508 )
c
c        ciorsq    square of refractive index
c
      implicit none
      logical  calcqe
      integer  numang,j
      real*8    gqsc, qext, qsca, xx, xmu(*)
      real*8 twothr,fivthr,sq,rtmp
      complex*16  a( 2 ), b( 2 ), cior, sforw, sback, s1(*), s2(*),
     $         tforw(*), tback(*)
c
      parameter  ( twothr = 2./3., fivthr = 5./3. )
      complex*16  ctmp, ciorsq
      sq( ctmp ) = dble( ctmp )**2 + dimag( ctmp )**2
c
c
      ciorsq = cior**2
      ctmp = dcmplx( 0.d0, twothr ) * ( ciorsq - 1.0 )
      a(1) = ctmp * ( 1.0 - 0.1 * xx**2 + (ciorsq/350. + 1./280.)*xx**4)
     $       / ( ciorsq + 2.0 + ( 1.0 - 0.7 * ciorsq ) * xx**2
     $           - ( ciorsq**2/175. - 0.275 * ciorsq + 0.25 ) * xx**4
     $           + xx**3 * ctmp * ( 1.0 - 0.1 * xx**2 ) )
c
      b(1) = (xx**2/30.) * ctmp * ( 1.0 + (ciorsq/35. - 1./14.) *xx**2 )
     $       / ( 1.0 - ( ciorsq/15. - 1./6. ) * xx**2 )
c
      a(2) = ( 0.1 * xx**2 ) * ctmp * ( 1.0 - xx**2 / 14. )
     $       / ( 2. * ciorsq + 3. - ( ciorsq/7. - 0.5 ) * xx**2 )
c
      qsca = 6. * xx**4 * ( sq(a(1)) + sq(b(1)) + fivthr * sq(a(2)) )
      gqsc = 6. * xx**4 * dble( a(1) * dconjg( a(2) + b(1) ) )
      qext = qsca
      if ( calcqe ) qext = 6. * xx * dble( a(1) + b(1) + fivthr * a(2) )
c
      rtmp = 1.5 * xx**3
      sforw      = rtmp * ( a(1) + b(1) + fivthr * a(2) )
      sback      = rtmp * ( a(1) - b(1) - fivthr * a(2) )
      tforw( 1 ) = rtmp * ( b(1) - fivthr * a(2) )
      tforw( 2 ) = rtmp * ( a(1) + 2. * fivthr * a(2) )
      tback( 1 ) = tforw( 1 )
      tback( 2 ) = rtmp * ( a(1) - 2. * fivthr * a(2) )
c
      do  10  j = 1, numang
         s1( j ) = rtmp * ( a(1) + ( b(1) + fivthr * a(2) ) * xmu(j) )
         s2( j ) = rtmp * ( b(1) + a(1) * xmu(j) + fivthr * a(2)
     $                      * ( 2. * xmu(j)**2 - 1. ) )
   10 continue
c                                     ** recover actual mie coefficients
      a( 1 ) = xx**3 * a( 1 )
      a( 2 ) = xx**3 * a( 2 )
      b( 1 ) = xx**3 * b( 1 )
      b( 2 ) = ( 0., 0. )
c
      return
      end
      subroutine  testmi ( qext, qsca, gqsc, sforw, sback, s1, s2,
     $                     tforw, tback, pmom, momdim, ok )
c
c         compare mie code test case results with correct answers
c         and return  ok=false  if even one result is inaccurate.
c
c         the test case is :  mie size parameter = 10
c                             refractive index   = 1.5 - 0.1 i
c                             scattering angle = 140 degrees
c                             1 sekera moment
c
c         results for this case may be found among the test cases
c         at the end of reference (1).
c
c         *** note *** when running on some computers, esp. in single
c         precision, the 'accur' criterion below may have to be relaxed.
c         however, if 'accur' must be set larger than 10**-3 for some
c         size parameters, your computer is probably not accurate
c         enough to do mie computations for those size parameters.
c
      implicit none
      integer momdim,m,n
      real*8    qext, qsca, gqsc, pmom( 0:momdim, * )
      complex*16  sforw, sback, s1(*), s2(*), tforw(*), tback(*)
      logical  ok, wrong
c
      real*8    accur, testqe, testqs, testgq, testpm( 0:1 )
      complex*16 testsf, testsb,tests1,tests2,testtf(2), testtb(2)
      data   testqe / 2.459791 /,  testqs / 1.235144 /,
     $       testgq / 1.139235 /,  testsf / ( 61.49476, -3.177994 ) /,
     $       testsb / ( 1.493434, 0.2963657 ) /,
     $       tests1 / ( -0.1548380, -1.128972) /,
     $       tests2 / ( 0.05669755, 0.5425681) /,
     $       testtf / ( 12.95238, -136.6436 ), ( 48.54238, 133.4656 ) /,
     $       testtb / ( 41.88414, -15.57833 ), ( 43.37758, -15.28196 )/,
     $       testpm / 227.1975, 183.6898 /
      real*8 calc,exact
c      data   accur / 1.e-5 /
      data   accur / 1.e-4 /
      wrong( calc, exact ) = dabs( (calc - exact) / exact ) .gt. accur
c
c
      ok = .true.
      if ( wrong( qext,testqe ) )
     $     call  tstbad( 'qext', abs((qext - testqe) / testqe), ok )
      if ( wrong( qsca,testqs ) )
     $     call  tstbad( 'qsca', abs((qsca - testqs) / testqs), ok )
      if ( wrong( gqsc,testgq ) )
     $     call  tstbad( 'gqsc', abs((gqsc - testgq) / testgq), ok )
c
      if ( wrong(  dble(sforw),  dble(testsf) ) .or.
     $     wrong( dimag(sforw), dimag(testsf) ) )
     $     call  tstbad( 'sforw', cdabs((sforw - testsf) / testsf), ok )
c
      if ( wrong(  dble(sback),  dble(testsb) ) .or.
     $     wrong( dimag(sback), dimag(testsb) ) )
     $     call  tstbad( 'sback', cdabs((sback - testsb) / testsb), ok )
c
      if ( wrong(  dble(s1(1)),  dble(tests1) ) .or.
     $     wrong( dimag(s1(1)), dimag(tests1) ) )
     $     call  tstbad( 's1', cdabs((s1(1) - tests1) / tests1), ok )
c
      if ( wrong(  dble(s2(1)),  dble(tests2) ) .or.
     $     wrong( dimag(s2(1)), dimag(tests2) ) )
     $     call  tstbad( 's2', cdabs((s2(1) - tests2) / tests2), ok )
c
      do  20  n = 1, 2
         if ( wrong(  dble(tforw(n)),  dble(testtf(n)) ) .or.
     $        wrong( dimag(tforw(n)), dimag(testtf(n)) ) )
     $        call  tstbad( 'tforw', cdabs( (tforw(n) - testtf(n)) /
     $                                     testtf(n) ), ok )
         if ( wrong(  dble(tback(n)),  dble(testtb(n)) ) .or.
     $        wrong( dimag(tback(n)), dimag(testtb(n)) ) )
     $        call  tstbad( 'tback', cdabs( (tback(n) - testtb(n)) /
     $                                      testtb(n) ), ok )
   20 continue
c
      do  30  m = 0, 1
         if ( wrong( pmom(m,1), testpm(m) ) )
     $        call  tstbad( 'pmom', dabs( (pmom(m,1)-testpm(m)) /
     $                                   testpm(m) ), ok )
   30 continue
c
      return
c
      end
      subroutine  errmsg( messag, fatal )
c
c        print out a warning or error message;  abort if error
c
      implicit none
      logical       fatal, once
      character*(*) messag
      integer       maxmsg, nummsg
      save          maxmsg, nummsg, once
      data nummsg / 0 /,  maxmsg / 100 /,  once / .false. /
c
c
      if ( fatal )  then
         write ( *, '(2a)' )  ' ******* error >>>>>>  ', messag
         stop
      end if
c
      nummsg = nummsg + 1
      if ( nummsg.gt.maxmsg )  then
         if ( .not.once )  write ( *,99 )
         once = .true.
      else
         write ( *, '(2a)' )  ' ******* warning >>>>>>  ', messag
      endif
c
      return
c
   99 format( ///,' >>>>>>  too many warning messages --  ',
     $   'they will no longer be printed  <<<<<<<', /// )
      end
      subroutine  wrtbad ( varnam, erflag )
c
c          write names of erroneous variables
c
c      input :   varnam = name of erroneous variable to be written
c                         ( character, any length )
c
c      output :  erflag = logical flag, set true by this routine
c ----------------------------------------------------------------------
      implicit none
      character*(*)  varnam
      logical        erflag
      integer        maxmsg, nummsg
      save  nummsg, maxmsg
      data  nummsg / 0 /,  maxmsg / 50 /
c
c
      nummsg = nummsg + 1
      write ( *, '(3a)' )  ' ****  input variable  ', varnam,
     $                     '  in error  ****'
      erflag = .true.
      if ( nummsg.eq.maxmsg )
     $   call  errmsg ( 'too many input errors.  aborting...$', .true. )
      return
c
      end
      subroutine  tstbad( varnam, relerr, ok )
c
c       write name (-varnam-) of variable failing self-test and its
c       percent error from the correct value.  return  ok = false.
c
      implicit none
      character*(*)  varnam
      logical        ok
      real*8          relerr
c
c
      ok = .false.
      write( *, '(/,3a,1p,e11.2,a)' )
     $       ' output variable  ', varnam,'  differed by', 100.*relerr,
     $       '  per cent from correct value.  self-test failed.'
      return
c
      end
