package chunktest

import "fmt"

const Version = 42

type User struct {
	Name string
}

func Add(a int, b int) int {
	return a + b
}

func (u *User) NameLength() int {
	return len(u.Name)
}

func (u *User) IsVIP() bool {
	return fmt.Sprintf("vip-%s", u.Name) == "vip"
}
