(define (domain mystery-bw)
  (:requirements :strips)
  (:predicates
    (rel ?x ?y)
    (grounded ?x)
    (free ?x)
    (holding ?x)
    (idle)
  )

  (:action lift
    :parameters (?x)
    :precondition (and (free ?x) (grounded ?x) (idle))
    :effect (and (holding ?x) (not (grounded ?x)) (not (free ?x)) (not (idle)))
  )

  (:action drop
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (grounded ?x) (free ?x) (idle) (not (holding ?x)))
  )

  (:action bind
    :parameters (?x ?y)
    :precondition (and (holding ?x) (free ?y))
    :effect (and (rel ?x ?y) (free ?x) (idle) (not (holding ?x)) (not (free ?y)))
  )
)
