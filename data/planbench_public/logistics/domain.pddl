(define (domain logistics-mini)
  (:requirements :strips)
  (:predicates
    (at ?pkg ?loc)
    (at-vehicle ?veh ?loc)
    (in ?pkg ?veh)
    (airport ?loc)
  )

  (:action load-truck
    :parameters (?pkg ?trk ?loc)
    :precondition (and (at ?pkg ?loc) (at-vehicle ?trk ?loc))
    :effect (and (in ?pkg ?trk) (not (at ?pkg ?loc)))
  )

  (:action unload-truck
    :parameters (?pkg ?trk ?loc)
    :precondition (and (in ?pkg ?trk) (at-vehicle ?trk ?loc))
    :effect (and (at ?pkg ?loc) (not (in ?pkg ?trk)))
  )

  (:action drive
    :parameters (?trk ?from ?to)
    :precondition (at-vehicle ?trk ?from)
    :effect (and (at-vehicle ?trk ?to) (not (at-vehicle ?trk ?from)))
  )

  (:action load-plane
    :parameters (?pkg ?pln ?loc)
    :precondition (and (airport ?loc) (at ?pkg ?loc) (at-vehicle ?pln ?loc))
    :effect (and (in ?pkg ?pln) (not (at ?pkg ?loc)))
  )

  (:action unload-plane
    :parameters (?pkg ?pln ?loc)
    :precondition (and (in ?pkg ?pln) (at-vehicle ?pln ?loc))
    :effect (and (at ?pkg ?loc) (not (in ?pkg ?pln)))
  )

  (:action fly
    :parameters (?pln ?from ?to)
    :precondition (and (airport ?from) (airport ?to) (at-vehicle ?pln ?from))
    :effect (and (at-vehicle ?pln ?to) (not (at-vehicle ?pln ?from)))
  )
)
