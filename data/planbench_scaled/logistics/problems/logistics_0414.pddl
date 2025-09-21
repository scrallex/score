(define (problem logistics_0414)
  (:domain logistics-mini)
  (:objects pkg1 pkg2 truck1 plane1 loc0 airportA airportB)
  (:init
    (airport airportA)
    (airport airportB)
    (at-vehicle truck1 loc0)
    (at-vehicle plane1 airportA)
    (at pkg1 loc0)
    (at pkg2 loc0)
  )
  (:goal
    (and
      (at pkg1 airportB)
      (at pkg2 airportB)
    )
  )
 )
