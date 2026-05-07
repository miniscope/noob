import type {TubeSpecification} from "../../src/types";

export const recursiveTube: TubeSpecification = {
    'noob_id': 'testing-recursive-parent',
    'noob_model': 'noob.tube.TubeSpecification',
    'noob_version': '0.0.1.dev155+g2f9639f',
    'input': {
        'parent_start': {'id': 'parent_start', 'type': 'int', 'scope': 'tube'},
        'parent_multiply': {'id': 'parent_multiply', 'type': 'int', 'scope': 'process'},
        'child_start': {'id': 'child_start', 'type': 'int', 'scope': 'tube'},
        'child_multiply': {'id': 'child_multiply', 'type': 'int', 'scope': 'process'}
    },
    'nodes': {
        'a': {'type': 'noob.testing.count_source', 'id': 'a', 'params': {'start': 'input.parent_start'}},
        'b': {
            'type': 'tube',
            'id': 'b',
            'depends': [{'child_multiply_inner': 'a.index'}, {'child_multiply_input': 'input.child_multiply'}],
            'params': {
                'tube': {
                    'noob_id': 'testing-recursive-child',
                    'noob_model': 'noob.tube.TubeSpecification',
                    'noob_version': '0.0.1.dev155+g2f9639f',
                    'input': {
                        'child_start': {'id': 'child_start', 'type_': 'int', 'scope':'tube' },
                        'child_multiply_inner': {'id': 'child_multiply_inner', 'type_': 'int', 'scope': 'process'},
                        'child_multiply_input': {'id': 'child_multiply_input', 'type_': 'int', 'scope':  'process'}
                    },
                    'nodes': {
                        'a': {'type': 'noob.testing.count_source', 'id': 'a', 'params': {'start': 'input.child_start'}},
                        'b': {'type': 'noob.testing.multiply', 'id': 'b', 'depends': [{'left': 'a.index'}, {'right': 'input.child_multiply_inner'}]},
                        'c': {'type': 'noob.testing.multiply', 'id': 'c', 'depends': [{'left': 'b.value'}, {'right': 'input.child_multiply_input'}]},
                        'd': {'type': 'return', 'id': 'd', 'depends': 'c.value'}
                    }
                }
            }
        },
        'c': {'type': 'noob.testing.multiply', 'id': 'c', 'depends': [{'left': 'b.value'}, {'right': 'input.parent_multiply'}]},
        'd': {'type': 'return', 'id': 'd', 'depends': [{'index': 'a.index'}, {'child': 'b.value'}, {'parent': 'c.value'}]}
    }
}
