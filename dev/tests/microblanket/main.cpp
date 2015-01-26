#include <assert.h>
#include <iostream>
#include <vector>
#include <microblanket.h>

using namespace std;
using namespace microblanket;

const int StackSize = 4*1024;

struct entry_t {
    unsigned fiber_id;
    enum {
        Enter,
        Middle,
        Exit
    } state;
};
vector<entry_t> entries;

struct MyFiber : public fiber_t<StackSize> {
    void run() {
        entries.push_back({fid, entry_t::Enter});
        yield();
        entries.push_back({fid, entry_t::Middle});
        yield();
        entries.push_back({fid, entry_t::Exit});
    }
};
typedef blanket_t<MyFiber> MyBlanket;

int main(int argc, const char **argv) {
    cout << "=== Starting" << endl;

    const unsigned nfibers = 50000;

    MyBlanket blanket{nfibers};

    for(unsigned it = 0; it < 10; it++) {
        blanket.clear();
        entries.clear();

        auto run = [](MyFiber *f){f->run();};

        for(unsigned i = 0; i < nfibers; i++) {
            blanket.spawn(run);
        }

        // Test iterator
        {
            unsigned i = 0;
            for(MyFiber &f: blanket) {
                assert(f.fid == i++);
            }
        }

        // Test [] operator
        {
            for(unsigned i = 0; i < nfibers; i++) {
                assert(blanket[i].fid == i);
            }
        }

        for(MyFiber &f: blanket) {
            f.resume();
        }
        for(MyFiber &f: blanket) {
            f.resume();
        }

        for(unsigned i = 0; i < nfibers; i++) {
            entry_t e = entries[i];
            assert(e.fiber_id == i);
            assert(e.state == entry_t::Enter);
        }
        for(unsigned i = 0; i < nfibers; i++) {
            entry_t e = entries[nfibers+i];
            assert(e.fiber_id == i);
            assert(e.state == entry_t::Middle);
        }
        for(unsigned i = 0; i < nfibers; i++) {
            entry_t e = entries[nfibers*2+i];
            assert(e.fiber_id == i);
            assert(e.state == entry_t::Exit);
        }
    }
    
    cout << "=== Done" << endl;

    return 0;
}
