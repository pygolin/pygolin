// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pygolin_runtime

import (
	"bytes"
	"fmt"
	"reflect"
	"sync/atomic"
	"unsafe"
)

var (
	// DictType is the object representing the Python 'dict' type.
	DictType              = newBasisType("dict", reflect.TypeOf(Dict{}), toDictUnsafe, ObjectType)
	dictItemIteratorType  = newBasisType("dictionary-itemiterator", reflect.TypeOf(dictItemIterator{}), toDictItemIteratorUnsafe, ObjectType)
	dictKeyIteratorType   = newBasisType("dictionary-keyiterator", reflect.TypeOf(dictKeyIterator{}), toDictKeyIteratorUnsafe, ObjectType)
	dictValueIteratorType = newBasisType("dictionary-valueiterator", reflect.TypeOf(dictValueIterator{}), toDictValueIteratorUnsafe, ObjectType)
)

const (
	// maxDictSize is the largest number of entries a dictionary can hold.
	// Dict sizes must be a power of two and this is the largest such
	// number multiplied by 2 is representable as uint32.
	maxDictSize            = 1 << 30
	minDictSize            = 4
	maxDictWithoutHashSize = 8
)

// dictEntry represents an element in array of entries of the dictTable of a Dict.
//
// hash and key field are immutable once set, therefore, they could be checked
// almost without synchronization (synchronization is made at dictTable level).
// They not overwritten even when entry were deleted. Deleted entry is detected
// by nil value.
//
// value is mutable, and therefore should be written and read with atomic
// (except insertAbsentEntry, since it is synced on Dict level).
// value is changed in a way nil->non-nil->non-nil->...->non-nil->nil. Once
// it is set to nil, it should not be overwritten again. It is not hard demand,
// but it simplifies a bit, and allows to preserve ordering, ie if dict had
// no key (since it were deleted) then after insertion it should be last.
type dictEntry struct {
	hash  int
	key   *Object
	value *Object
}

// dictTable is the hash table underlying Dict.
// It preserves insertion order of key by using array of entries and indices
// into array in a hash part.
// For concurrency, indices and entries pointers are never overwritten.
// When entries array is full, new dictTable is created.
// Hash part is twice larger than entries array, therefore, there is no need
// to trace its fill factor separately.
// Index to array is 1 based, and zero means never-written empty element.
// Index is not cleared when entry deleted, but could be overwritten when
// same key inserted again.
// Index should be written with atomic after new entry is written (except insertAbsentEntry),
// therefore lookupEntry could rely on it for synchronization.
// fill field also plays important role in synchronization: it should be incremented
// with atomic after new entry is written (except insertAbsentEntry). Using this
// fact, and entry.key immutability, iteration could access key without synchronization,
// though it still should check entry.value using atomic.
type dictTable struct {
	// used is real number of alive values in a table
	used uint32
	// fill is the number of slots that are used or once were used but have
	// since been cleared. Thus used <= fill <= len(entries).
	// it is like len(entries) if entries were slice
	fill uint32
	// capa is a real capacity of entries, ie cap(entries) if entries were slice
	capa uint32
	// mask is len(indicies)-1 , and as we use power-of-two tables, it is
	// used for index calculation: hash&mask == hash%len(indices)
	mask uint32
	// indices is a hash part of dictTable. It contains indices into entries array.
	indices *[maxDictSize * 2]uint32
	// entries is a an array of entries. It is used to keep insertion order of
	// entries.
	entries *[maxDictSize]dictEntry
}

// newDictTable allocates a table where at least minCapacity entries can be
// accommodated. minCapacity must be <= maxDictSize.
func newDictTable(numEntries uint32) *dictTable {
	// It rounds required to nearest power of two using bit trick if neccessary.
	// It doesn't increase capacity if it is already power of two, unless it is
	// less than minDictSize.
	// For tables smaller than maxDictWithoutHashSize indices array is not allocated.
	if numEntries <= minDictSize {
		numEntries = minDictSize
	} else if (numEntries-1)&numEntries != 0 {
		numEntries |= numEntries >> 1
		numEntries |= numEntries >> 2
		numEntries |= numEntries >> 4
		numEntries |= numEntries >> 8
		numEntries |= numEntries >> 16
		numEntries++
	}
	t := &dictTable{
		capa:    numEntries,
		entries: (*[maxDictSize]dictEntry)(unsafe.Pointer(&make([]dictEntry, numEntries)[0])),
	}
	if numEntries > maxDictWithoutHashSize {
		t.mask = numEntries*2 - 1
		t.indices = (*[maxDictSize * 2]uint32)(unsafe.Pointer(&make([]uint32, numEntries*2)[0]))
	}
	return t
}

func (t *dictTable) loadIndex(i uint32) uint32 {
	return atomic.LoadUint32(&t.indices[i])
}

func (t *dictTable) storeIndex(i, idx uint32) {
	atomic.StoreUint32(&t.indices[i], idx)
}

func (t *dictTable) loadUsed() int {
	if t == nil {
		return 0
	}
	return int(atomic.LoadUint32(&t.used))
}

func (t *dictTable) incUsed(n int) {
	atomic.AddUint32(&t.used, uint32(n))
}

func (t *dictTable) loadFill() uint32 {
	if t == nil {
		return 0
	}
	return atomic.LoadUint32(&t.fill)
}

func (t *dictTable) incFill(n int) {
	atomic.AddUint32(&t.fill, uint32(n))
}

func (t *dictEntry) loadValue() *Object {
	p := (*unsafe.Pointer)(unsafe.Pointer(&t.value))
	return (*Object)(atomic.LoadPointer(p))
}

func (t *dictEntry) storeValue(o *Object) {
	p := (*unsafe.Pointer)(unsafe.Pointer(&t.value))
	atomic.StorePointer(p, unsafe.Pointer(o))
}

func (t *dictEntry) swapValue(o *Object) *Object {
	p := (*unsafe.Pointer)(unsafe.Pointer(&t.value))
	return (*Object)(atomic.SwapPointer(p, unsafe.Pointer(o)))
}

// insertAbsentEntry adds the populated entry to t assuming that the key
// specified in entry is absent from t. Since the key is absent, no key
// comparisons are necessary to perform the insert.
// It doesn't use atomic instructions and there fore it should operate only on
// non-yet reachable table. Dict.storeTable is used as synchronization point.
func (t *dictTable) insertAbsentEntry(entry *dictEntry) {
	if t.fill == t.capa {
		panic("overrun")
	}
	if mask := t.mask; mask != 0 {
		i := uint32(entry.hash) & mask
		perturb := uint(entry.hash)
		index := i
		// The key we're trying to insert is known to be absent from the dict
		// so probe for the first zero entry.
		for ; t.indices[index] != 0; index = i & mask {
			i, perturb = dictNextIndex(i, perturb)
		}
		t.indices[index] = t.fill + 1
	}
	t.entries[t.fill] = *entry
	t.used++
	t.fill++
}

// lookupEntry returns the index to indices table and entry in entries with the given hash and key.
// Non-nil entry could be returned if entry were deleted, therefore entry.value should be checked after.
// When t.indices is allocated, it uses synchronization on t.fill, otherwise, it uses synchronization
// on index (since it is written atomically after entry is written).
// Therefore it is not necessary to lock the dict to do entry lookups in a consistent way.
// When entry is not found, returned index points into empty place in t.indices.
func (t *dictTable) lookupEntry(f *Frame, hash int, key *Object) (uint32, *dictEntry, *BaseException) {
	if t == nil {
		return 0, nil, nil
	}
	mask := t.mask
	if mask == 0 {
		// need to iterate in reverse order to find inserted-after-deleted entries
		for eidx := t.loadFill(); eidx > 0; eidx-- {
			entry := &t.entries[eidx-1]
			if entry.hash == hash {
				o, raised := Eq(f, entry.key, key)
				if raised != nil {
					return 0, nil, raised
				}
				eq, raised := IsTrue(f, o)
				if raised != nil {
					return 0, nil, raised
				}
				if eq {
					return 0, entry, nil
				}
			}
		}
		return 0, nil, nil
	}
	i, perturb := uint32(hash)&mask, uint(hash)
	index := i & mask
	for {
		idx := t.loadIndex(index)
		if idx == 0 {
			return index, nil, nil
		}
		eidx := idx - 1
		entry := &t.entries[eidx]
		if entry.hash == hash {
			o, raised := Eq(f, entry.key, key)
			if raised != nil {
				return 0, nil, raised
			}
			eq, raised := IsTrue(f, o)
			if raised != nil {
				return 0, nil, raised
			}
			if eq {
				return index, entry, nil
			}
		}
		i, perturb = dictNextIndex(i, perturb)
		index = i & mask
	}
}

// writeNewEntry writes entry at the end of entries array, and its index
// into position, returned by previously called lookupEntry (so, index
// points into empty position or position bounded with the same key).
// It differs from insertAbsentEntry because a) index position is known,
// b) it has to use atomics to store index and increment t.fill.
func (t *dictTable) writeNewEntry(index uint32, nentry *dictEntry) {
	eidx := t.fill
	t.entries[eidx] = *nentry
	if t.mask != 0 {
		// store index atomically after entry is stored to synchronize
		// with lookupEntry
		t.storeIndex(index, eidx+1)
	}
	t.incUsed(1)
	// increment fill atomically after entry is stored to synchronize
	// with iteration.
	t.incFill(1)
}

// writeValue rewrites value in non-empty entry.
// Old value should be non-nil and therefore it is not checked.
// While it is not hard demand, but it is part of insertion-order keeping.
func (t *dictTable) writeValue(entry *dictEntry, value *Object) {
	entry.storeValue(value)
	if value == nil {
		t.incUsed(-1)
	}
}

// growTable allocates table at least twice larger than already used space.
// It should be called when t.fill == t.capa .
func (t *dictTable) growTable() (*dictTable, bool) {
	if t == nil {
		// allocate minimal table
		return newDictTable(0), true
	}
	var n uint32
	if t.used < t.capa/2 {
		n = t.used * 2
	} else if t.capa <= maxDictSize/2 {
		n = t.capa * 2
	} else {
		return nil, false
	}
	newTable := newDictTable(n)
	for i := uint32(0); i < t.capa; i++ {
		oldEntry := &t.entries[i]
		if oldEntry.value != nil {
			newTable.insertAbsentEntry(oldEntry)
		}
	}
	return newTable, true
}

// dictEntryIterator is used to iterate over the entries in a dictTable in an
// arbitrary order.
type dictEntryIterator struct {
	index uint32
	table *dictTable
}

// newDictEntryIterator creates a dictEntryIterator object for d. It assumes
// that d.mutex is held by the caller.
func newDictEntryIterator(d *Dict) dictEntryIterator {
	return dictEntryIterator{table: d.loadTable()}
}

// next advances this iterator to the next occupied entry and returns it.
// it returns nil, nil when there is no more elements.
func (iter *dictEntryIterator) next() (*Object, *Object) {
	filled := iter.table.loadFill()
	for {
		index := atomic.AddUint32(&iter.index, 1) - 1
		if index >= filled {
			atomic.AddUint32(&iter.index, ^uint32(0))
			return nil, nil
		}
		entry := &iter.table.entries[index]
		if value := entry.loadValue(); value != nil {
			return entry.key, value
		}
	}
}

// dictVersionGuard is used to detect when a dict has been modified.
type dictVersionGuard struct {
	dict    *Dict
	version int64
}

func newDictVersionGuard(d *Dict) dictVersionGuard {
	return dictVersionGuard{d, d.loadVersion()}
}

// check returns false if the dict held by g has changed since g was created,
// true otherwise.
func (g *dictVersionGuard) check() bool {
	return g.dict.loadVersion() == g.version
}

// Dict represents Python 'dict' objects. The public methods of *Dict are
// thread safe.
type Dict struct {
	Object
	table *dictTable
	// We use a recursive mutex for synchronization because the hash and
	// key comparison operations may re-enter DelItem/SetItem.
	mutex recursiveMutex
	// version is incremented whenever the Dict is modified. See:
	// https://www.python.org/dev/peps/pep-0509/
	version int64
}

// NewDict returns an empty Dict.
func NewDict() *Dict {
	return &Dict{Object: Object{typ: DictType}}
}

func newStringDict(items map[string]*Object) *Dict {
	if len(items) > maxDictSize {
		panic(fmt.Sprintf("dictionary too big: %d", len(items)))
	}
	table := newDictTable(uint32(len(items)))
	for key, value := range items {
		table.insertAbsentEntry(&dictEntry{hashString(key), NewStr(key).ToObject(), value})
	}
	return &Dict{Object: Object{typ: DictType}, table: table}
}

func toDictUnsafe(o *Object) *Dict {
	return (*Dict)(o.toPointer())
}

// loadTable atomically loads and returns d's underlying dictTable.
func (d *Dict) loadTable() *dictTable {
	p := (*unsafe.Pointer)(unsafe.Pointer(&d.table))
	return (*dictTable)(atomic.LoadPointer(p))
}

// storeTable atomically updates d's underlying dictTable to the one given.
func (d *Dict) storeTable(table *dictTable) {
	p := (*unsafe.Pointer)(unsafe.Pointer(&d.table))
	atomic.StorePointer(p, unsafe.Pointer(table))
}

// loadVersion atomically loads and returns d's version.
func (d *Dict) loadVersion() int64 {
	// 64bit atomic ops need to be 8 byte aligned. This compile time check
	// verifies alignment by creating a negative constant for an unsigned type.
	// See sync/atomic docs for details.
	const blank = -(unsafe.Offsetof(d.version) % 8)
	return atomic.LoadInt64(&d.version)
}

// incVersion atomically increments d's version.
func (d *Dict) incVersion() {
	// 64bit atomic ops need to be 8 byte aligned. This compile time check
	// verifies alignment by creating a negative constant for an unsigned type.
	// See sync/atomic docs for details.
	const blank = -(unsafe.Offsetof(d.version) % 8)
	atomic.AddInt64(&d.version, 1)
}

// DelItem removes the entry associated with key from d. It returns true if an
// item was removed, or false if it did not exist in d.
func (d *Dict) DelItem(f *Frame, key *Object) (bool, *BaseException) {
	originValue, raised := d.putItem(f, key, nil, true)
	if raised != nil {
		return false, raised
	}
	return originValue != nil, nil
}

// DelItemString removes the entry associated with key from d. It returns true
// if an item was removed, or false if it did not exist in d.
func (d *Dict) DelItemString(f *Frame, key string) (bool, *BaseException) {
	return d.DelItem(f, NewStr(key).ToObject())
}

// GetItem looks up key in d, returning the associated value or nil if key is
// not present in d.
func (d *Dict) GetItem(f *Frame, key *Object) (*Object, *BaseException) {
	hash, raised := Hash(f, key)
	if raised != nil {
		return nil, raised
	}
	_, entry, raised := d.loadTable().lookupEntry(f, hash.Value(), key)
	if raised != nil {
		return nil, raised
	}
	if entry != nil {
		return entry.loadValue(), nil
	}
	return nil, nil
}

// GetItemString looks up key in d, returning the associated value or nil if
// key is not present in d.
func (d *Dict) GetItemString(f *Frame, key string) (*Object, *BaseException) {
	return d.GetItem(f, NewStr(key).ToObject())
}

// Pop looks up key in d, returning and removing the associalted value if exist,
// or nil if key is not present in d.
func (d *Dict) Pop(f *Frame, key *Object) (*Object, *BaseException) {
	return d.putItem(f, key, nil, true)
}

// Keys returns a list containing all the keys in d.
func (d *Dict) Keys(f *Frame) *List {
	table := d.loadTable()
	fill := int(table.loadFill())
	used := table.loadUsed()
	// since `used` is loaded after `fill`, then number of alive values
	// in t.entries[:fill] could not be larger than `used`
	keys := make([]*Object, used)
	i := 0
	for k := 0; k < fill; k++ {
		entry := &table.entries[k]
		if value := entry.loadValue(); value != nil {
			keys[i] = entry.key
			i++
		}
	}
	return NewList(keys[:i]...)
}

// Len returns the number of entries in d.
func (d *Dict) Len() int {
	return d.loadTable().loadUsed()
}

// putItem associates value with key in d, returning the old associated value if
// the key was added, or nil if it was not already present in d.
func (d *Dict) putItem(f *Frame, key, value *Object, overwrite bool) (*Object, *BaseException) {
	hash, raised := Hash(f, key)
	if raised != nil {
		return nil, raised
	}
	hashv := hash.Value()
	d.mutex.Lock(f)
	// we do not use `defer d.mutex.Unlock(f)` here because defer is not free: it slows putItem by 30% .
	// Since putItem is a hot place, lets Unlock manually.
	t := d.table
	v := d.version
	index, entry, raised := t.lookupEntry(f, hashv, key)
	if raised != nil {
		d.mutex.Unlock(f)
		return nil, raised
	}
	if v != d.version {
		// Dictionary was recursively modified. Blow up instead
		// of trying to recover.
		d.mutex.Unlock(f)
		return nil, f.RaiseType(RuntimeErrorType, "dictionary changed during write")
	}
	var originValue *Object
	if entry == nil || entry.value == nil {
		// either key were never inserted, or it was deleted
		if value != nil {
			if t == nil || t.fill == d.table.capa {
				if newTable, ok := d.table.growTable(); ok {
					newTable.insertAbsentEntry(&dictEntry{
						hash:  hashv,
						key:   key,
						value: value,
					})
					// synchronization point
					d.storeTable(newTable)
				} else {
					d.mutex.Unlock(f)
					return nil, f.RaiseType(OverflowErrorType, errResultTooLarge)
				}
			} else {
				t.writeNewEntry(index, &dictEntry{
					hash:  hashv,
					key:   key,
					value: value,
				})
			}
			d.incVersion()
		}
	} else {
		originValue = entry.value
		if overwrite {
			t.writeValue(entry, value)
			d.incVersion()
			if value == nil && t.used < t.capa/8 && t.fill > t.capa/8*5 {
				if newTable, ok := t.growTable(); ok {
					d.storeTable(newTable)
					// doesn't increment version here, because we didn't change content in growTable.
				} else {
					d.mutex.Unlock(f)
					panic("some unknown error on downsizing dictionary")
				}
			}
		}
	}
	d.mutex.Unlock(f)
	return originValue, raised
}

// SetItem associates value with key in d.
func (d *Dict) SetItem(f *Frame, key, value *Object) *BaseException {
	_, raised := d.putItem(f, key, value, true)
	return raised
}

// SetItemString associates value with key in d.
func (d *Dict) SetItemString(f *Frame, key string, value *Object) *BaseException {
	return d.SetItem(f, NewStr(key).ToObject(), value)
}

// ToObject upcasts d to an Object.
func (d *Dict) ToObject() *Object {
	return &d.Object
}

// Update copies the items from the mapping or sequence of 2-tuples o into d.
func (d *Dict) Update(f *Frame, o *Object) (raised *BaseException) {
	var iter *Object
	if o.isInstance(DictType) {
		d2 := toDictUnsafe(o)
		d2.mutex.Lock(f)
		// Concurrent modifications to d2 will cause Update to raise
		// "dictionary changed during iteration".
		iter = newDictItemIterator(d2).ToObject()
		d2.mutex.Unlock(f)
	} else {
		iter, raised = Iter(f, o)
	}
	if raised != nil {
		return raised
	}
	return seqForEach(f, iter, func(item *Object) *BaseException {
		return seqApply(f, item, func(elems []*Object, _ bool) *BaseException {
			if numElems := len(elems); numElems != 2 {
				format := "dictionary update sequence element has length %d; 2 is required"
				return f.RaiseType(ValueErrorType, fmt.Sprintf(format, numElems))
			}
			return d.SetItem(f, elems[0], elems[1])
		})
	})
}

// dictsAreEqual returns true if d1 and d2 have the same keys and values, false
// otherwise. If either d1 or d2 are concurrently modified then RuntimeError is
// raised.
func dictsAreEqual(f *Frame, d1, d2 *Dict) (bool, *BaseException) {
	if d1 == d2 {
		return true, nil
	}
	// Do not hold both locks at the same time to avoid deadlock.
	d1.mutex.Lock(f)
	iter := newDictEntryIterator(d1)
	g1 := newDictVersionGuard(d1)
	len1 := d1.Len()
	d1.mutex.Unlock(f)
	d2.mutex.Lock(f)
	g2 := newDictVersionGuard(d2)
	len2 := d2.Len()
	d2.mutex.Unlock(f)
	if len1 != len2 {
		return false, nil
	}
	result := true
	for key, value := iter.next(); key != nil && result; key, value = iter.next() {
		if v, raised := d2.GetItem(f, key); raised != nil {
			return false, raised
		} else if v == nil {
			result = false
		} else {
			eq, raised := Eq(f, value, v)
			if raised != nil {
				return false, raised
			}
			result, raised = IsTrue(f, eq)
			if raised != nil {
				return false, raised
			}
		}
	}
	if !g1.check() || !g2.check() {
		return false, f.RaiseType(RuntimeErrorType, "dictionary changed during iteration")
	}
	return result, nil
}

func dictClear(f *Frame, args Args, _ KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "clear", args, DictType); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(args[0])
	d.mutex.Lock(f)
	d.table = newDictTable(0)
	d.incVersion()
	d.mutex.Unlock(f)
	return None, nil
}

func dictContains(f *Frame, seq, value *Object) (*Object, *BaseException) {
	item, raised := toDictUnsafe(seq).GetItem(f, value)
	if raised != nil {
		return nil, raised
	}
	return GetBool(item != nil).ToObject(), nil
}

func dictCopy(f *Frame, args Args, _ KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "copy", args, DictType); raised != nil {
		return nil, raised
	}
	return DictType.Call(f, args, nil)
}

func dictDelItem(f *Frame, o, key *Object) *BaseException {
	deleted, raised := toDictUnsafe(o).DelItem(f, key)
	if raised != nil {
		return raised
	}
	if !deleted {
		return raiseKeyError(f, key)
	}
	return nil
}

func dictEq(f *Frame, v, w *Object) (*Object, *BaseException) {
	if !w.isInstance(DictType) {
		return NotImplemented, nil
	}
	eq, raised := dictsAreEqual(f, toDictUnsafe(v), toDictUnsafe(w))
	if raised != nil {
		return nil, raised
	}
	return GetBool(eq).ToObject(), nil
}

func dictGet(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	expectedTypes := []*Type{DictType, ObjectType, ObjectType}
	argc := len(args)
	if argc == 2 {
		expectedTypes = expectedTypes[:2]
	}
	if raised := checkMethodArgs(f, "get", args, expectedTypes...); raised != nil {
		return nil, raised
	}
	item, raised := toDictUnsafe(args[0]).GetItem(f, args[1])
	if raised == nil && item == nil {
		item = None
		if argc > 2 {
			item = args[2]
		}
	}
	return item, raised
}

func dictHasKey(f *Frame, args Args, _ KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "has_key", args, DictType, ObjectType); raised != nil {
		return nil, raised
	}
	return dictContains(f, args[0], args[1])
}

func dictItems(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "items", args, DictType); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(args[0])
	d.mutex.Lock(f)
	iter := newDictItemIterator(d).ToObject()
	d.mutex.Unlock(f)
	return ListType.Call(f, Args{iter}, nil)
}

func dictIterItems(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "iteritems", args, DictType); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(args[0])
	d.mutex.Lock(f)
	iter := newDictItemIterator(d).ToObject()
	d.mutex.Unlock(f)
	return iter, nil
}

func dictIterKeys(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "iterkeys", args, DictType); raised != nil {
		return nil, raised
	}
	return dictIter(f, args[0])
}

func dictIterValues(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "itervalues", args, DictType); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(args[0])
	d.mutex.Lock(f)
	iter := newDictValueIterator(d).ToObject()
	d.mutex.Unlock(f)
	return iter, nil
}

func dictKeys(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "keys", args, DictType); raised != nil {
		return nil, raised
	}
	return toDictUnsafe(args[0]).Keys(f).ToObject(), nil
}

func dictGetItem(f *Frame, o, key *Object) (*Object, *BaseException) {
	item, raised := toDictUnsafe(o).GetItem(f, key)
	if raised != nil {
		return nil, raised
	}
	if item == nil {
		return nil, raiseKeyError(f, key)
	}
	return item, nil
}

func dictInit(f *Frame, o *Object, args Args, kwargs KWArgs) (*Object, *BaseException) {
	var expectedTypes []*Type
	argc := len(args)
	if argc > 0 {
		expectedTypes = []*Type{ObjectType}
	}
	if raised := checkFunctionArgs(f, "__init__", args, expectedTypes...); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(o)
	if argc > 0 {
		if raised := d.Update(f, args[0]); raised != nil {
			return nil, raised
		}
	}
	for _, kwarg := range kwargs {
		if raised := d.SetItemString(f, kwarg.Name, kwarg.Value); raised != nil {
			return nil, raised
		}
	}
	return None, nil
}

func dictIter(f *Frame, o *Object) (*Object, *BaseException) {
	d := toDictUnsafe(o)
	d.mutex.Lock(f)
	iter := newDictKeyIterator(d).ToObject()
	d.mutex.Unlock(f)
	return iter, nil
}

func dictLen(f *Frame, o *Object) (*Object, *BaseException) {
	d := toDictUnsafe(o)
	ret := NewInt(d.Len()).ToObject()
	return ret, nil
}

func dictNE(f *Frame, v, w *Object) (*Object, *BaseException) {
	if !w.isInstance(DictType) {
		return NotImplemented, nil
	}
	eq, raised := dictsAreEqual(f, toDictUnsafe(v), toDictUnsafe(w))
	if raised != nil {
		return nil, raised
	}
	return GetBool(!eq).ToObject(), nil
}

func dictNew(f *Frame, t *Type, _ Args, _ KWArgs) (*Object, *BaseException) {
	d := toDictUnsafe(newObject(t))
	d.table = newDictTable(0)
	return d.ToObject(), nil
}

func dictPop(f *Frame, args Args, _ KWArgs) (*Object, *BaseException) {
	expectedTypes := []*Type{DictType, ObjectType, ObjectType}
	argc := len(args)
	if argc == 2 {
		expectedTypes = expectedTypes[:2]
	}
	if raised := checkMethodArgs(f, "pop", args, expectedTypes...); raised != nil {
		return nil, raised
	}
	key := args[1]
	d := toDictUnsafe(args[0])
	item, raised := d.Pop(f, key)
	if raised == nil && item == nil {
		if argc > 2 {
			item = args[2]
		} else {
			raised = raiseKeyError(f, key)
		}
	}
	return item, raised
}

func dictPopItem(f *Frame, args Args, _ KWArgs) (item *Object, raised *BaseException) {
	if raised := checkMethodArgs(f, "popitem", args, DictType); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(args[0])
	d.mutex.Lock(f)
	defer d.mutex.Unlock(f)
	if d.table.used == 0 {
		return nil, f.RaiseType(KeyErrorType, "popitem(): dictionary is empty")
	}
	// unfortunately, 3.7 standardized popping last key-value
	for i := int(d.table.fill) - 1; i >= 0; i-- {
		entry := &d.table.entries[i]
		if entry.value != nil {
			item = NewTuple(entry.key, entry.value).ToObject()
			entry.storeValue(nil)
			d.table.incUsed(-1)
			d.incVersion()
			return item, nil
		}
	}
	panic("there shall be at least one item")
}

func dictRepr(f *Frame, o *Object) (*Object, *BaseException) {
	d := toDictUnsafe(o)
	if f.reprEnter(d.ToObject()) {
		return NewStr("{...}").ToObject(), nil
	}
	defer f.reprLeave(d.ToObject())
	// Lock d so that we get a consistent view of it. Otherwise we may
	// return a state that d was never actually in.
	d.mutex.Lock(f)
	defer d.mutex.Unlock(f)
	var buf bytes.Buffer
	buf.WriteString("{")
	iter := newDictEntryIterator(d)
	i := 0
	for key, value := iter.next(); key != nil; key, value = iter.next() {
		if i > 0 {
			buf.WriteString(", ")
		}
		s, raised := Repr(f, key)
		if raised != nil {
			return nil, raised
		}
		buf.WriteString(s.Value())
		buf.WriteString(": ")
		if s, raised = Repr(f, value); raised != nil {
			return nil, raised
		}
		buf.WriteString(s.Value())
		i++
	}
	buf.WriteString("}")
	return NewStr(buf.String()).ToObject(), nil
}

func dictSetDefault(f *Frame, args Args, _ KWArgs) (*Object, *BaseException) {
	argc := len(args)
	if argc == 1 {
		return nil, f.RaiseType(TypeErrorType, "setdefault expected at least 1 arguments, got 0")
	}
	if argc > 3 {
		return nil, f.RaiseType(TypeErrorType, fmt.Sprintf("setdefault expected at most 2 arguments, got %v", argc-1))
	}
	expectedTypes := []*Type{DictType, ObjectType, ObjectType}
	if argc == 2 {
		expectedTypes = expectedTypes[:2]
	}
	if raised := checkMethodArgs(f, "setdefault", args, expectedTypes...); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(args[0])
	key := args[1]
	var value *Object
	if argc > 2 {
		value = args[2]
	} else {
		value = None
	}
	originValue, raised := d.putItem(f, key, value, false)
	if originValue != nil {
		return originValue, raised
	}
	return value, raised
}

func dictSetItem(f *Frame, o, key, value *Object) *BaseException {
	return toDictUnsafe(o).SetItem(f, key, value)
}

func dictUpdate(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	expectedTypes := []*Type{DictType, ObjectType}
	argc := len(args)
	if argc == 1 {
		expectedTypes = expectedTypes[:1]
	}
	if raised := checkMethodArgs(f, "update", args, expectedTypes...); raised != nil {
		return nil, raised
	}
	d := toDictUnsafe(args[0])
	if argc > 1 {
		if raised := d.Update(f, args[1]); raised != nil {
			return nil, raised
		}
	}
	for _, kwarg := range kwargs {
		if raised := d.SetItemString(f, kwarg.Name, kwarg.Value); raised != nil {
			return nil, raised
		}
	}
	return None, nil
}

func dictValues(f *Frame, args Args, kwargs KWArgs) (*Object, *BaseException) {
	if raised := checkMethodArgs(f, "values", args, DictType); raised != nil {
		return nil, raised
	}
	iter, raised := dictIterValues(f, args, nil)
	if raised != nil {
		return nil, raised
	}
	return ListType.Call(f, Args{iter}, nil)
}

func initDictType(dict map[string]*Object) {
	dict["clear"] = newBuiltinFunction("clear", dictClear).ToObject()
	dict["copy"] = newBuiltinFunction("copy", dictCopy).ToObject()
	dict["get"] = newBuiltinFunction("get", dictGet).ToObject()
	dict["has_key"] = newBuiltinFunction("has_key", dictHasKey).ToObject()
	dict["items"] = newBuiltinFunction("items", dictItems).ToObject()
	dict["iteritems"] = newBuiltinFunction("iteritems", dictIterItems).ToObject()
	dict["iterkeys"] = newBuiltinFunction("iterkeys", dictIterKeys).ToObject()
	dict["itervalues"] = newBuiltinFunction("itervalues", dictIterValues).ToObject()
	dict["keys"] = newBuiltinFunction("keys", dictKeys).ToObject()
	dict["pop"] = newBuiltinFunction("pop", dictPop).ToObject()
	dict["popitem"] = newBuiltinFunction("popitem", dictPopItem).ToObject()
	dict["setdefault"] = newBuiltinFunction("setdefault", dictSetDefault).ToObject()
	dict["update"] = newBuiltinFunction("update", dictUpdate).ToObject()
	dict["values"] = newBuiltinFunction("values", dictValues).ToObject()
	DictType.slots.Contains = &binaryOpSlot{dictContains}
	DictType.slots.DelItem = &delItemSlot{dictDelItem}
	DictType.slots.Eq = &binaryOpSlot{dictEq}
	DictType.slots.GetItem = &binaryOpSlot{dictGetItem}
	DictType.slots.Hash = &unaryOpSlot{hashNotImplemented}
	DictType.slots.Init = &initSlot{dictInit}
	DictType.slots.Iter = &unaryOpSlot{dictIter}
	DictType.slots.Len = &unaryOpSlot{dictLen}
	DictType.slots.NE = &binaryOpSlot{dictNE}
	DictType.slots.New = &newSlot{dictNew}
	DictType.slots.Repr = &unaryOpSlot{dictRepr}
	DictType.slots.SetItem = &setItemSlot{dictSetItem}
}

type dictItemIterator struct {
	Object
	iter  dictEntryIterator
	guard dictVersionGuard
}

// newDictItemIterator creates a dictItemIterator object for d. It assumes that
// d.mutex is held by the caller.
func newDictItemIterator(d *Dict) *dictItemIterator {
	return &dictItemIterator{
		Object: Object{typ: dictItemIteratorType},
		iter:   newDictEntryIterator(d),
		guard:  newDictVersionGuard(d),
	}
}

func toDictItemIteratorUnsafe(o *Object) *dictItemIterator {
	return (*dictItemIterator)(o.toPointer())
}

func (iter *dictItemIterator) ToObject() *Object {
	return &iter.Object
}

func dictItemIteratorIter(f *Frame, o *Object) (*Object, *BaseException) {
	return o, nil
}

func dictItemIteratorNext(f *Frame, o *Object) (ret *Object, raised *BaseException) {
	iter := toDictItemIteratorUnsafe(o)
	key, value, raised := dictIteratorNext(f, &iter.iter, &iter.guard)
	if raised != nil {
		return nil, raised
	}
	return NewTuple2(key, value).ToObject(), nil
}

func initDictItemIteratorType(map[string]*Object) {
	dictItemIteratorType.flags &^= typeFlagBasetype | typeFlagInstantiable
	dictItemIteratorType.slots.Iter = &unaryOpSlot{dictItemIteratorIter}
	dictItemIteratorType.slots.Next = &unaryOpSlot{dictItemIteratorNext}
}

type dictKeyIterator struct {
	Object
	iter  dictEntryIterator
	guard dictVersionGuard
}

// newDictKeyIterator creates a dictKeyIterator object for d. It assumes that
// d.mutex is held by the caller.
func newDictKeyIterator(d *Dict) *dictKeyIterator {
	return &dictKeyIterator{
		Object: Object{typ: dictKeyIteratorType},
		iter:   newDictEntryIterator(d),
		guard:  newDictVersionGuard(d),
	}
}

func toDictKeyIteratorUnsafe(o *Object) *dictKeyIterator {
	return (*dictKeyIterator)(o.toPointer())
}

func (iter *dictKeyIterator) ToObject() *Object {
	return &iter.Object
}

func dictKeyIteratorIter(f *Frame, o *Object) (*Object, *BaseException) {
	return o, nil
}

func dictKeyIteratorNext(f *Frame, o *Object) (*Object, *BaseException) {
	iter := toDictKeyIteratorUnsafe(o)
	key, _, raised := dictIteratorNext(f, &iter.iter, &iter.guard)
	if raised != nil {
		return nil, raised
	}
	return key, nil
}

func initDictKeyIteratorType(map[string]*Object) {
	dictKeyIteratorType.flags &^= typeFlagBasetype | typeFlagInstantiable
	dictKeyIteratorType.slots.Iter = &unaryOpSlot{dictKeyIteratorIter}
	dictKeyIteratorType.slots.Next = &unaryOpSlot{dictKeyIteratorNext}
}

type dictValueIterator struct {
	Object
	iter  dictEntryIterator
	guard dictVersionGuard
}

// newDictValueIterator creates a dictValueIterator object for d. It assumes
// that d.mutex is held by the caller.
func newDictValueIterator(d *Dict) *dictValueIterator {
	return &dictValueIterator{
		Object: Object{typ: dictValueIteratorType},
		iter:   newDictEntryIterator(d),
		guard:  newDictVersionGuard(d),
	}
}

func toDictValueIteratorUnsafe(o *Object) *dictValueIterator {
	return (*dictValueIterator)(o.toPointer())
}

func (iter *dictValueIterator) ToObject() *Object {
	return &iter.Object
}

func dictValueIteratorIter(f *Frame, o *Object) (*Object, *BaseException) {
	return o, nil
}

func dictValueIteratorNext(f *Frame, o *Object) (*Object, *BaseException) {
	iter := toDictValueIteratorUnsafe(o)
	_, value, raised := dictIteratorNext(f, &iter.iter, &iter.guard)
	if raised != nil {
		return nil, raised
	}
	return value, nil
}

func initDictValueIteratorType(map[string]*Object) {
	dictValueIteratorType.flags &^= typeFlagBasetype | typeFlagInstantiable
	dictValueIteratorType.slots.Iter = &unaryOpSlot{dictValueIteratorIter}
	dictValueIteratorType.slots.Next = &unaryOpSlot{dictValueIteratorNext}
}

func raiseKeyError(f *Frame, key *Object) *BaseException {
	s, raised := ToStr(f, key)
	if raised == nil {
		raised = f.RaiseType(KeyErrorType, s.Value())
	}
	return raised
}

func dictNextIndex(i uint32, perturb uint) (uint32, uint) {
	return (i << 2) + i + uint32(perturb) + 1, perturb >> 5
}

func dictIteratorNext(f *Frame, iter *dictEntryIterator, guard *dictVersionGuard) (*Object, *Object, *BaseException) {
	// NOTE: The behavior here diverges from CPython where an iterator that
	// is exhausted will always return StopIteration regardless whether the
	// underlying dict is subsequently modified. In Grumpy, an iterator for
	// a dict that has been modified will always raise RuntimeError even if
	// the iterator was exhausted before the modification.
	key, value := iter.next()
	if !guard.check() {
		return nil, nil, f.RaiseType(RuntimeErrorType, "dictionary changed during iteration")
	}
	if key == nil {
		return nil, nil, f.Raise(StopIterationType.ToObject(), nil, nil)
	}
	return key, value, nil
}
